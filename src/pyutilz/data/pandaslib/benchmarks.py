# ----------------------------------------------------------------------------------------------------------------------------
# Dataframe compression benchmarks
# ----------------------------------------------------------------------------------------------------------------------------
"""Dataframe read/write & compression benchmarks.

Split out of the historical flat ``pyutilz.data.pandaslib`` module; re-exported
from the package ``__init__`` to preserve the public import surface.
"""

from ._common import (
    gc,
    np,
    pd,
    warnings,
    join,
    getsize,
    ensure_dir_exists,
    timer,
    chain,
    tempfile,
    shutil,
    tqdmu,
    logger,
)

import pyutilz.data.pandaslib as _facade  # patchable-name indirection for internal cross-calls

from .dtypes import get_df_memory_consumption
from .frames import remove_constant_columns


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
            read_times, write_times, read_sizes, write_sizes = _facade.measure_read_write_performance(
                df=df,
                fname=fname,
                read_method="read_parquet",
                read_params=dict(**engine_params),
                write_method=write_method,
                write_params=dict(**engine_params, compression=compr),
                nrepeats=nrepeats,
            )

            _facade.pack_benchmark_results(res, config, read_times, write_times, read_sizes, write_sizes)

            for compression_level in range(1, max_compression_level + 1):
                config = f"{file_format}-{engine}-{compr}-{compression_level}"

                fname = join(temp_folder, rf"{config}.{file_format}")
                try:
                    read_times, write_times, read_sizes, write_sizes = _facade.measure_read_write_performance(
                        df=df,
                        fname=fname,
                        read_method="read_parquet",
                        read_params=dict(**engine_params),
                        write_method=write_method,
                        write_params=dict(**engine_params, compression=compr, compression_level=compression_level),
                        nrepeats=nrepeats,
                    )

                    _facade.pack_benchmark_results(res, config, read_times, write_times, read_sizes, write_sizes)
                except Exception:
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
        read_times, write_times, read_sizes, write_sizes = _facade.measure_read_write_performance(
            df=df,
            fname=fname,
            read_method="read_pickle",
            read_params=dict(compression={"method": compr}),
            write_method="to_pickle",
            write_params=dict(compression={"method": compr}, protocol=-1),  # "compresslevel": level
            nrepeats=nrepeats,
        )

        _facade.pack_benchmark_results(res, config, read_times, write_times, read_sizes, write_sizes)


def benchmark_dataframe_hdf_compression(res, temp_folder, df, nrepeats):
    file_format = "hdf"

    for level in tqdmu(range(1, 10), desc=f"{file_format} engine", leave=False):
        for compr in tqdmu("zlib lzo bzip2 blosc".split(), desc=f"{file_format} compression method", leave=False):
            config = f"{file_format}-{compr}"  # -{level}

            fname = join(temp_folder, rf"{config}.{file_format}.{compr}")
            read_times, write_times, read_sizes, write_sizes = _facade.measure_read_write_performance(
                df=df,
                fname=fname,
                read_method="read_hdf",
                read_params=dict(complib=compr),
                write_method="to_hdf",
                write_params=dict(complib=compr, complevel=level, key="test"),
                nrepeats=nrepeats,
            )

            _facade.pack_benchmark_results(res, config, read_times, write_times, read_sizes, write_sizes)


def benchmark_dataframe_csv_compression(res, temp_folder, df, nrepeats):
    file_format = "csv"

    for compr in tqdmu(["zip", "gzip", "bz2", "zstd", "xz", "tar"], desc=f"{file_format} compression method", leave=False):
        config = f"{file_format}-{compr}"

        fname = join(temp_folder, rf"{config}.{file_format}.{compr}")
        read_times, write_times, read_sizes, write_sizes = _facade.measure_read_write_performance(
            df=df,
            fname=fname,
            read_method="read_csv",
            read_params=dict(compression={"method": compr}),
            write_method="to_csv",
            write_params=dict(compression={"method": compr}),
            nrepeats=nrepeats,
        )

        _facade.pack_benchmark_results(res, config, read_times, write_times, read_sizes, write_sizes)


def benchmark_dataframe_orc_compression(res, temp_folder, df, nrepeats):
    file_format = "orc"

    config = f"{file_format}"

    fname = join(temp_folder, rf"{config}.{file_format}")
    read_times, write_times, read_sizes, write_sizes = _facade.measure_read_write_performance(
        df=df,
        fname=fname,
        read_method="read_orc",
        read_params=dict(),
        write_method="to_orc",
        write_params=dict(),
        nrepeats=nrepeats,
    )

    _facade.pack_benchmark_results(res, config, read_times, write_times, read_sizes, write_sizes)


def benchmark_dataframe_feather_compression(res, temp_folder, df, nrepeats):
    file_format = "feather"

    config = f"{file_format}"

    fname = join(temp_folder, rf"{config}.{file_format}")
    read_times, write_times, read_sizes, write_sizes = _facade.measure_read_write_performance(
        df=df,
        fname=fname,
        read_method="read_feather",
        read_params=dict(),
        write_method="to_feather",
        write_params=dict(),
        nrepeats=nrepeats,
    )

    _facade.pack_benchmark_results(res, config, read_times, write_times, read_sizes, write_sizes)


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
        logger.info("Pandas: %s, DF size: %.2fMb, Dtypes: %s", pd.__version__, df_size, df.dtypes.value_counts().to_dict())

    if benchmark_dir_path:
        ensure_dir_exists(benchmark_dir_path)
    temp_folder = tempfile.mkdtemp(dir=benchmark_dir_path)

    res = []

    for func in (
        _facade.benchmark_dataframe_feather_compression,
        _facade.benchmark_dataframe_orc_compression,
        _facade.benchmark_dataframe_hdf_compression,
        _facade.benchmark_dataframe_pickle_compression,
        _facade.benchmark_dataframe_csv_compression,
    ):
        try:
            func(res, temp_folder, df, nrepeats)
        except Exception as e:
            logger.error(e)

    # Parquet has different signature, handle separately
    try:
        parquet_results = _facade.benchmark_dataframe_parquet_compression(df, temp_folder, nrepeats)
        res.extend(parquet_results.to_dict("records"))
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

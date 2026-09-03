# ----------------------------------------------------------------------------------------------------------------------------
# Dataframe compression benchmarks
# ----------------------------------------------------------------------------------------------------------------------------
"""Dataframe read/write & compression benchmarks.

Split out of the historical flat ``pyutilz.data.pandaslib`` module; re-exported
from the package ``__init__`` to preserve the public import surface.
"""

from ._common import (
    Any,
    List,
    Optional,
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
import pyutilz.data.pandaslib as _facade  # patchable-name indirection for internal cross-calls

from .dtypes import get_df_memory_consumption
from .frames import remove_constant_columns


def measure_read_write_performance(df: pd.DataFrame, fname: str, read_method: str, read_params: dict, write_method: str, write_params: dict, nrepeats: int):
    """Write & read ``df`` to/from ``fname`` ``nrepeats`` times using the given pandas methods.

    Returns 4 ``np.ndarray`` arrays (read_times, write_times, read_sizes, write_sizes),
    one value per repeat; sizes are in megabytes (write: on-disk file size, read: in-memory
    dataframe size).
    """
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
    """Append a summary row (config name + mean/std of each timing/size array) to ``res`` in place."""
    res.append([config, *list(chain(*[(np.mean(arr), np.std(arr)) for arr in (read_times, write_times, read_sizes, write_sizes)]))])


def benchmark_dataframe_parquet_compression(
    df: pd.DataFrame,
    temp_folder: str,
    nrepeats: int = 3,
    engines: tuple = ("fastparquet", "pyarrow"),
    max_compression_level: int = 22,
    skip_configs: tuple = ("parquet-fastparquet-brotli",),
    write_method: str = "to_parquet",
) -> pd.DataFrame:
    """Benchmark parquet read/write across engines, compression codecs and (for supported codecs) compression levels.

    Returns a dataframe with one row per ``{engine}-{codec}[-{level}]`` config and mean/std
    read/write time & size columns (``config``, ``mean_read_time``, ``std_read_time``,
    ``mean_write_time``, ``std_write_time``, ``mean_read_size``, ``std_read_size``,
    ``mean_write_size``, ``std_write_size``). Configs listed in ``skip_configs`` are skipped; per-level
    configs that error out are logged and skipped rather than aborting the whole benchmark.
    """
    res: List[Any] = []
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

    return pd.DataFrame(  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime
        res,
        columns=[
            "config",
            # Singular, matching the column vocabulary benchmark_dataframe_compression assembles its
            # combined frame with -- these rows are appended straight into it.
            *"mean_read_time,std_read_time,mean_write_time,std_write_time,mean_read_size,std_read_size,mean_write_size,std_write_size".split(","),
        ],
    )


def _run_compression_sweep(res, temp_folder, df, nrepeats, file_format, sweep):
    """Run one file_format's read/write compression sweep and append each config's results to
    ``res`` in place. ``sweep`` is an iterable of ``(compr_label, read_method, read_params,
    write_method, write_params)`` tuples; an empty ``compr_label`` means a single default
    config (no compression suffix in the config name/filename).

    2026-08-02 near-duplicate-function-body finding: the pickle/hdf/csv/orc/feather compression
    benchmarks independently duplicated this file-naming + measure + pack step around their own
    per-config loop; only the per-format list of configs to sweep now differs between them.
    """
    for compr_label, read_method, read_params, write_method, write_params in sweep:
        config = f"{file_format}-{compr_label}" if compr_label else file_format
        suffix = f".{compr_label}" if compr_label else ""
        fname = join(temp_folder, rf"{config}.{file_format}{suffix}")
        read_times, write_times, read_sizes, write_sizes = _facade.measure_read_write_performance(
            df=df,
            fname=fname,
            read_method=read_method,
            read_params=read_params,
            write_method=write_method,
            write_params=write_params,
            nrepeats=nrepeats,
        )
        _facade.pack_benchmark_results(res, config, read_times, write_times, read_sizes, write_sizes)


def _standard_method_compression_sweep(file_format: str, read_method: str, write_method: str, extra_write_params: Optional[dict] = None):
    """Shared ``(compr, read_method, read_params, write_method, write_params)`` generator for formats
    (pickle, csv) whose read/write methods both accept a ``compression={"method": compr}`` kwarg over
    the standard zip/gzip/bz2/zstd/xz/tar sweep -- factored out of the near-identical
    ``benchmark_dataframe_pickle_compression``/``benchmark_dataframe_csv_compression`` bodies that
    duplicated this generator verbatim aside from the method names and one extra pickle-only kwarg.
    """
    write_params_extra = extra_write_params or {}
    for compr in tqdmu(["zip", "gzip", "bz2", "zstd", "xz", "tar"], desc=f"{file_format} compression method", leave=False):
        yield (compr, read_method, dict(compression={"method": compr}), write_method, dict(compression={"method": compr}, **write_params_extra))


def benchmark_dataframe_pickle_compression(res, temp_folder, df, nrepeats):
    """Benchmark pickle read/write across compression methods (zip/gzip/bz2/zstd/xz/tar), appending each config's results to ``res`` in place."""
    file_format = "pickle"
    sweep = _standard_method_compression_sweep(file_format, "read_pickle", "to_pickle", extra_write_params={"protocol": -1})
    _run_compression_sweep(res, temp_folder, df, nrepeats, file_format, sweep)


def benchmark_dataframe_hdf_compression(res, temp_folder, df, nrepeats):
    """Benchmark HDF5 read/write across compression libraries (zlib/lzo/bzip2/blosc) and compression levels, appending each config's results to ``res`` in place."""
    file_format = "hdf"

    def sweep():
        """Yield one (compr, read_method, read_params, write_method, write_params) tuple per (level, compr) combination."""
        for level in tqdmu(range(1, 10), desc=f"{file_format} engine", leave=False):
            for compr in tqdmu("zlib lzo bzip2 blosc".split(), desc=f"{file_format} compression method", leave=False):
                yield compr, "read_hdf", dict(complib=compr), "to_hdf", dict(complib=compr, complevel=level, key="test")

    _run_compression_sweep(res, temp_folder, df, nrepeats, file_format, sweep())


def benchmark_dataframe_csv_compression(res, temp_folder, df, nrepeats):
    """Benchmark CSV read/write across compression methods (zip/gzip/bz2/zstd/xz/tar), appending each config's results to ``res`` in place."""
    file_format = "csv"
    sweep = _standard_method_compression_sweep(file_format, "read_csv", "to_csv")
    _run_compression_sweep(res, temp_folder, df, nrepeats, file_format, sweep)


def benchmark_dataframe_orc_compression(res, temp_folder, df, nrepeats):
    """Benchmark ORC read/write (single default config), appending the results to ``res`` in place."""
    _run_compression_sweep(res, temp_folder, df, nrepeats, "orc", [("", "read_orc", dict(), "to_orc", dict())])


def benchmark_dataframe_feather_compression(res, temp_folder, df, nrepeats):
    """Benchmark feather read/write (single default config), appending the results to ``res`` in place."""
    _run_compression_sweep(res, temp_folder, df, nrepeats, "feather", [("", "read_feather", dict(), "to_feather", dict())])


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

    res: Any = []

    for func in (
        _facade.benchmark_dataframe_feather_compression,
        _facade.benchmark_dataframe_orc_compression,
        _facade.benchmark_dataframe_hdf_compression,
        _facade.benchmark_dataframe_pickle_compression,
        _facade.benchmark_dataframe_csv_compression,
    ):
        try:
            func(res, temp_folder, df, nrepeats)
        except Exception as e:  # noqa: PERF203 -- per-iteration fault isolation is intentional (one benchmark failing shouldn't skip the rest)
            logger.error(e)

    # Parquet has different signature, handle separately
    try:
        parquet_results = _facade.benchmark_dataframe_parquet_compression(df, temp_folder, nrepeats)
        # Rows, not records: `res` holds LISTS (see pack_benchmark_results), and pd.DataFrame() renders a
        # dict inside a list-of-rows as its KEYS, so every parquet row used to come out as a row of the
        # literal strings "config", "mean_read_times", ... -- which the very next sort_values() then
        # compared against floats and died on with an unrelated-looking numpy UFuncTypeError.
        res.extend(parquet_results.itertuples(index=False, name=None))
    except Exception as e:
        logger.error(e)

    if should_clean_temp_folder:
        shutil.rmtree(temp_folder)

    res = (
        pd.DataFrame(res, columns=["config", *chain(*[("mean_" + arr, "std_" + arr) for arr in "read_time write_time read_size write_size".split()])])
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

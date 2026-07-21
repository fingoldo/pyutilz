"""Dataframe IO helpers: load/read, concat-and-flush, multi-file merging and
pyarrow parquet reads.

Split out of the historical flat ``pyutilz.data.pandaslib`` module; re-exported
from the package ``__init__`` to preserve the public import surface.
"""

from ._common import (
    gc,
    os,
    pd,
    glob,
    join,
    sep,
    tqdmu,
    logger,
)

import pyutilz.data.pandaslib as _facade  # patchable-name indirection (see below)

from .dtypes import optimize_dtypes, get_df_memory_consumption
from typing import Callable, Optional


def load_df(fpath: str, tail: int) -> pd.DataFrame:
    """
    Load a pickled dataframe from ``fpath``, optionally keeping only the last ``tail`` rows.

    Args:
        fpath: path to the pickled dataframe file.
        tail: if a positive int, the dataframe is truncated to its last ``tail`` rows; falsy/None leaves it untouched.
    """
    logger.info("Loading data from file %s...", fpath)

    df = pd.read_pickle(fpath)
    if tail is not None:
        if tail > 0:
            logger.info("Limiting to last %s responses", tail)
            df = df.tail(tail)
    return df


def concat_and_flush_df_list(
    lst: list, file_name: str, to_csv: bool = False, csv_cols: Optional[list] = None, write_fcn: str = "to_pickle", write_extension: str = "pckl", set_index: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Concatenate all dataframes in ``lst`` into one, clear ``lst`` in place to free memory, then write and return the result.

    If ``to_csv`` is True the joined dataframe is written as CSV (header+overwrite when ``csv_cols`` is None,
    else appending only ``csv_cols`` without header); otherwise it is optionally indexed by ``set_index`` and
    written via ``getattr(joined_df, write_fcn)`` to ``f"{file_name}.{write_extension}"``. Returns None if ``lst`` is empty.
    """
    if len(lst) > 0:
        joined_df = pd.concat(lst, axis=0, ignore_index=True)
        lst.clear()
        gc.collect()
        if to_csv:
            if csv_cols is None:
                joined_df.to_csv(f"{file_name}.csv", mode="w", header=True)
            else:
                joined_df[csv_cols].to_csv(f"{file_name}.csv", mode="a", header=False)
            return joined_df
        else:
            if set_index:
                joined_df.set_index(set_index, inplace=True)
            getattr(joined_df, write_fcn)(f"{file_name}.{write_extension}")
            return joined_df
    return None


def read_stats_from_multiple_files(
    joint_file_name: str = "joint_features",
    folder: str = "features",
    max_files: int = 250,
    template: str = "*.pckl",
    exclude: Optional[str] = None,
    read_fcn: str = "read_pickle",
    write_fcn: str = "to_pickle",
    write_extension: str = "pckl",
    delete_after: bool = False,
    sentinel_field: Optional[str] = None,
    sentinel_fcn: Optional[Callable] = None,
    set_index: Optional[str] = None,
    optimize: bool = False,
    save_on_successful_optimization: bool = False,
    min_size_improvement_percent: float = 0.05,
    min_size_improvement: float = 5.0,
):
    """
    Read and merge dataframes from files matching ``template`` in ``folder`` (up to ``max_files``), then flush the merge to disk.

    Each matching file (read via ``read_fcn``, skipping any whose name contains ``exclude``) is optionally dtype-optimized
    via ``optimize_dtypes`` and re-saved with ``write_fcn``/``write_extension`` when the optimization shrinks it by at least
    ``min_size_improvement_percent`` or ``min_size_improvement`` GB (only if ``save_on_successful_optimization``). If
    ``sentinel_field`` is given, a column with that name (auto-suffixed with "1" if already present) is added to each
    frame, holding either the source filename or ``sentinel_fcn(filename)``. Collected frames are merged via
    ``concat_and_flush_df_list`` into ``joint_file_name`` (optionally indexed by ``set_index``); source files are
    deleted afterwards if ``delete_after`` is True. Returns the merged dataframe, or None if nothing was merged.
    """
    lst = []
    fnames = []
    for _i, filename in tqdmu(enumerate(glob.glob(join(folder, template)))):
        if exclude:
            if exclude in filename:
                continue

        fnames.append(filename)
        tmp_df = getattr(pd, read_fcn)(filename)

        # deep=True (byte-accurate for object/string columns) -- a shallow .memory_usage() call
        # undercounts object-dtype columns by 30x+ (pointer size only), making the resave-trigger
        # comparison below unreliable for exactly the string-heavy dataframes this function
        # batch-processes.
        old_size = get_df_memory_consumption(tmp_df, deep=True) / 1024**3
        logger.info("Merging %s with %s rows of size %.1f Gb", filename, len(tmp_df), old_size)

        if optimize:
            tmp_df = optimize_dtypes(tmp_df)
            gc.collect()
            new_size = get_df_memory_consumption(tmp_df, deep=True) / 1024**3
            logger.info("After optimization, %s got size %.1f Gb", filename, new_size)

            if save_on_successful_optimization:
                if new_size <= old_size * (1 - min_size_improvement_percent) or old_size - new_size >= min_size_improvement:
                    logger.info("Re-saving file %s due to lower size", filename)
                    getattr(tmp_df, write_fcn)(f"{'.'.join(filename.split('.')[:-1])}.{write_extension}")

        if sentinel_field:
            # Use local copy to avoid mutation persisting across files
            current_sentinel = sentinel_field
            while current_sentinel in tmp_df:
                logger.warning(f"Sentinel field {current_sentinel} was already in the frame {filename}")
                current_sentinel += "1"

            fname_part = filename.split(sep)[-1]
            if sentinel_fcn:
                tmp_df[current_sentinel] = sentinel_fcn(fname_part)
            else:
                tmp_df[current_sentinel] = fname_part

        lst.append(tmp_df)
        if max_files is not None:
            if len(lst) >= max_files:
                break
        del tmp_df
    if len(lst) > 0:
        try:
            res = _facade.concat_and_flush_df_list(lst, file_name=joint_file_name, write_fcn=write_fcn, write_extension=write_extension, set_index=set_index)
            if res is not None:
                logger.info("Final df size (%s rows)", len(res))
            if delete_after:
                for _i, filename in enumerate(fnames):
                    try:
                        os.remove(filename)
                    except Exception as e:  # nosec B110 - best-effort cleanup of already-concatenated source files; a failed remove (e.g. permissions/in-use) shouldn't abort the already-computed result  # noqa: PERF203 -- per-iteration fault isolation is intentional
                        logger.debug("Failed to remove source file %s after concat: %s", filename, e)
            return res
        except Exception as e:  # nosec B110 - best-effort concat-and-flush across the file list; on failure the function simply returns None (no result to salvage) rather than crashing the caller
            logger.debug("Failed to concat and flush df list: %s", e)


def read_parquet_with_pyarrow(path: str, nrows: int) -> pd.DataFrame:
    """
    Read a parquet dataset at ``path`` via pyarrow and return it as a pandas dataframe.

    If ``nrows`` is truthy, only the first ``nrows`` rows are scanned and returned; otherwise the whole dataset is read.
    """
    if nrows:
        df = _facade.dataset(path).scanner().head(nrows).to_pandas()
    else:
        df = _facade.dataset(path).scanner().to_pandas()

    return df

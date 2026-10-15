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
import pyutilz.data.pandaslib as _facade  # patchable-name indirection (see below)

from .dtypes import optimize_dtypes, get_df_memory_consumption
from typing import Callable, Optional, Tuple


def load_df(fpath: str, tail: int) -> pd.DataFrame:
    """
    Load a pickled dataframe from ``fpath``, optionally keeping only the last ``tail`` rows.

    Args:
        fpath: path to the pickled dataframe file.
        tail: if a positive int, the dataframe is truncated to its last ``tail`` rows; falsy/None leaves it untouched.

    Returns:
        the loaded dataframe, truncated to its last ``tail`` rows when ``tail`` is a positive int.
    """
    logger.info("Loading data from file %s...", fpath)

    df = pd.read_pickle(fpath)
    if tail is not None:
        if tail > 0:
            logger.info("Limiting to last %s responses", tail)
            df = df.tail(tail)
    return df


def concat_and_flush_df_list(
    lst: list,
    file_name: str,
    to_csv: bool = False,
    csv_cols: Optional[list] = None,
    write_fcn: str = "to_pickle",
    write_extension: str = "pckl",
    set_index: Optional[str] = None,
    ignore_index: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Concatenate all dataframes in ``lst`` into one, clear ``lst`` in place to free memory, then write and return the result.

    If ``to_csv`` is True the joined dataframe is written as CSV (header+overwrite when ``csv_cols`` is None,
    else appending only ``csv_cols`` without header); otherwise it is optionally indexed by ``set_index`` and
    written via ``getattr(joined_df, write_fcn)`` to ``f"{file_name}.{write_extension}"``. Returns None if ``lst`` is empty.

    ``ignore_index`` defaults to True, which DISCARDS whatever index the input frames carried and replaces
    it with a fresh RangeIndex -- an index that was not also a column (a timestamp index, an entity id) is
    unrecoverable afterwards, and ``set_index`` cannot bring it back because it can only promote a column.
    Pass False to keep the inputs' index labels in the concatenated result and in the written file.
    """
    if len(lst) > 0:
        joined_df = pd.concat(lst, axis=0, ignore_index=ignore_index)
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


def _optimize_and_maybe_resave(
    tmp_df: pd.DataFrame,
    filename: str,
    old_size: float,
    write_fcn: str,
    write_extension: str,
    save_on_successful_optimization: bool,
    min_size_improvement_percent: float,
    min_size_improvement: float,
) -> pd.DataFrame:
    """Dtype-optimize ``tmp_df``, re-saving the source file if the size shrink passes the thresholds."""
    tmp_df = optimize_dtypes(tmp_df)
    gc.collect()
    new_size = get_df_memory_consumption(tmp_df, deep=True) / 1024**3
    logger.info("After optimization, %s got size %.1f Gb", filename, new_size)

    if save_on_successful_optimization:
        if new_size <= old_size * (1 - min_size_improvement_percent) or old_size - new_size >= min_size_improvement:
            logger.info("Re-saving file %s due to lower size", filename)
            getattr(tmp_df, write_fcn)(f"{'.'.join(filename.split('.')[:-1])}.{write_extension}")
    return tmp_df


def _add_sentinel_column(tmp_df: pd.DataFrame, filename: str, sentinel_field: str, sentinel_fcn: Optional[Callable]) -> None:
    """Add a column named ``sentinel_field`` (suffixed until unique) holding the source filename or ``sentinel_fcn(filename)``."""
    # Use local copy to avoid mutation persisting across files
    current_sentinel = sentinel_field
    while current_sentinel in tmp_df:
        logger.warning(f"Sentinel field {current_sentinel} was already in the frame {filename}")  # audit-exempt: unthrottled_hot_loop_log -- per sentinel collision, which is typically zero or one per call
        current_sentinel += "1"

    fname_part = filename.split(sep)[-1]
    if sentinel_fcn:
        tmp_df[current_sentinel] = sentinel_fcn(fname_part)
    else:
        tmp_df[current_sentinel] = fname_part


def _collect_frames(
    folder: str,
    template: str,
    max_files: int,
    exclude: Optional[str],
    read_fcn: str,
    optimize: bool,
    save_on_successful_optimization: bool,
    min_size_improvement_percent: float,
    min_size_improvement: float,
    write_fcn: str,
    write_extension: str,
    sentinel_field: Optional[str],
    sentinel_fcn: Optional[Callable],
) -> Tuple[list, list]:
    """Read matching files (in order, up to ``max_files``) and return their frames plus the file names they came from."""
    lst = []
    fnames = []
    for _i, filename in tqdmu(enumerate(glob.glob(join(folder, template)))):
        if exclude:
            if exclude in filename:
                continue

        fnames.append(filename)
        tmp_df = getattr(pd, read_fcn)(filename)

        # deep=True (byte-accurate for object/string columns) is what the resave-trigger comparison in
        # _optimize_and_maybe_resave needs: a shallow memory_usage() undercounts object-dtype columns by
        # 30x+. Deep accounting visits every object cell (~0.3-1 s per 1M-row string frame), so it is paid
        # only when the optimisation decision actually uses it; the plain log line uses the shallow figure.
        if optimize:
            old_size = get_df_memory_consumption(tmp_df, deep=True) / 1024**3
        else:
            old_size = get_df_memory_consumption(tmp_df, deep=False) / 1024**3
        logger.info("Merging %s with %s rows of size %.1f Gb", filename, len(tmp_df), old_size)

        if optimize:
            tmp_df = _optimize_and_maybe_resave(
                tmp_df, filename, old_size, write_fcn, write_extension, save_on_successful_optimization, min_size_improvement_percent, min_size_improvement
            )

        if sentinel_field:
            _add_sentinel_column(tmp_df, filename, sentinel_field, sentinel_fcn)

        lst.append(tmp_df)
        if max_files is not None:
            if len(lst) >= max_files:
                break
        del tmp_df
    return lst, fnames


def _try_remove_source(filename: str) -> bool:
    """Remove one already-concatenated source file. False when it stays on disk (permissions, in use); logged at DEBUG."""
    try:
        os.remove(filename)
        return True
    except Exception as e:  # nosec B110 - best-effort cleanup of already-concatenated source files; the caller reports what stayed behind
        logger.warning("Failed to remove source file %s after concat: %s", filename, e)
        return False


def _flush_and_cleanup(
    lst: list,
    fnames: list,
    joint_file_name: str,
    write_fcn: str,
    write_extension: str,
    set_index: Optional[str],
    delete_after: bool,
) -> Optional[pd.DataFrame]:
    """Concatenate and flush ``lst`` to ``joint_file_name``, then optionally delete the source files.

    A failed concat-and-flush is re-raised: the caller must not be able to read a write failure as "nothing merged".
    A failed source-file removal is not fatal (the merged result exists), but it is logged at WARNING with the count
    of files left behind.
    """
    res = _facade.concat_and_flush_df_list(lst, file_name=joint_file_name, write_fcn=write_fcn, write_extension=write_extension, set_index=set_index)
    if res is not None:
        logger.info("Final df size (%s rows)", len(res))
    if delete_after:
        left_behind = [filename for filename in fnames if not _try_remove_source(filename)]
        if left_behind:
            logger.warning("%d source file(s) could not be removed after the concat and are still on disk, e.g. %s", len(left_behind), left_behind[0])
    return res


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
) -> Optional[pd.DataFrame]:
    """
    Read and merge dataframes from files matching ``template`` in ``folder`` (up to ``max_files``), then flush the merge to disk.

    Each matching file (read via ``read_fcn``, skipping any whose name contains ``exclude``) is optionally dtype-optimized
    via ``optimize_dtypes`` and re-saved with ``write_fcn``/``write_extension`` when the optimization shrinks it by at least
    ``min_size_improvement_percent`` or ``min_size_improvement`` GB (only if ``save_on_successful_optimization``). If
    ``sentinel_field`` is given, a column with that name (auto-suffixed with "1" if already present) is added to each
    frame, holding either the source filename or ``sentinel_fcn(filename)``. Collected frames are merged via
    ``concat_and_flush_df_list`` into ``joint_file_name`` (optionally indexed by ``set_index``); source files are
    deleted afterwards if ``delete_after`` is True. Returns the merged dataframe, or None if nothing was merged.
    A failed concat-and-flush raises; it is never reported as None.
    """
    lst, fnames = _collect_frames(
        folder,
        template,
        max_files,
        exclude,
        read_fcn,
        optimize,
        save_on_successful_optimization,
        min_size_improvement_percent,
        min_size_improvement,
        write_fcn,
        write_extension,
        sentinel_field,
        sentinel_fcn,
    )
    if len(lst) > 0:
        # A concat-and-flush failure propagates out of _flush_and_cleanup; None here means only "nothing collected".
        return _flush_and_cleanup(lst, fnames, joint_file_name, write_fcn, write_extension, set_index, delete_after)
    return None


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

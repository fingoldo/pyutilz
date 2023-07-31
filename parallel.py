# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from .pythonlib import ensure_installed

ensure_installed("psutil pandas")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

import heapq
import psutil
import pandas as pd
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool, set_start_method
from joblib import Parallel, delayed, parallel_backend
import contextlib

from .system import tqdmu


def split_list_into_chunks(the_list: list, chunk_size: int) -> list:
    """
    >>>list(split_list_into_chunks(list(range(10)),3))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    """
    t = len(the_list)
    n = int(t / chunk_size)
    for i in range(n + 1):
        l = i * chunk_size
        r = l + chunk_size
        if r > t:
            r = t
        if r>l:
            yield the_list[l:r]


def split_list_into_chunks_indices(the_list: list, chunk_size: int) -> list:
    """
    >>>list(split_list_into_chunks(list(range(10)),3))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    """
    t = len(the_list)
    n = int(t / chunk_size)
    for i in range(n + 1):
        l = i * chunk_size
        r = l + chunk_size
        if r > t:
            r = t
        yield l, r


def split_list_into_nchunks_indices(the_list: list, nchunks: int) -> list:
    """
    >>>list(split_list_into_nchunks_indices(list(range(10)),3))
    [(0, 3), (3, 6), (6, 10)]
    """
    total_length = len(the_list)
    step = total_length // nchunks
    for chunk in range(nchunks):
        if chunk == nchunks - 1:
            l = step * chunk
            r = total_length

            yield l, r
        else:
            yield step * chunk, (chunk + 1) * step


def split_array(arr: object, step: int) -> list:
    """
    Returns list of (a,b) tuples of array split into chunks using certain step size.
    >>>split_array(np.random.uniform(0,1,5477),step=1000)
    [(0, 1000),
     (1000, 2000),
     (2000, 3000),
     (3000, 4000),
     (4000, 5000),
     (5000, 5477)]
    """

    l = len(arr)
    a = 0
    b = a
    res = []
    assert step > 0
    while True:
        b = a + step
        if b >= l:
            b = l
        if b > 0:
            res.append((a, b))
        if b == l:
            break
        a = b
    return res


def distribute_work(workload: Sequence, nworkers: int) -> tuple:
    """Distribute array workload into nworkers chunks of approximately same total size."""
    planned_work_per_worker = [[] for _ in range(nworkers)]
    workload_indices_per_worker = [[] for _ in range(nworkers)]
    totals = [(0, i) for i in range(nworkers)]
    heapq.heapify(totals)
    for i, value in enumerate(workload):
        total, index = heapq.heappop(totals)
        planned_work_per_worker[index].append(value)
        workload_indices_per_worker[index].append(i)
        heapq.heappush(totals, (total + value, index))
    return planned_work_per_worker, workload_indices_per_worker


def parallel_run(
    jobslist: Sequence,
    n_jobs: int = -1,
    backend: str = None,
    max_nbytes:int=50_000, 
    verbose: int = 0,
    **parallel_kwargs
):
    """Runs function in parallel using the joblib package with flexible backend (including Dask)."""

    ctx_mgr = parallel_backend(backend) if (backend and "dask" in backend) else contextlib.nullcontext()
    with ctx_mgr:
        return Parallel(n_jobs=n_jobs, backend=None if (backend and "dask" in backend) else backend,max_nbytes=max_nbytes, verbose=verbose,**parallel_kwargs)(jobslist)


def applyfunc_parallel(
    iterable: list,
    func: object,
    n_cores: int = None,
    return_dataframe: bool = True,
    logical: bool = False,
    initializer=None,
    initargs=(),
    use_threads: bool = False,
) -> list:
    """Runs function in parallel using the multiprocessing Pool."""
    if n_cores is None:
        n_cores = min(psutil.cpu_count(logical=logical), len(iterable))
    try:
        fname = func.__name__
    except:
        fname = "function"

    logger.info(f"Applying of {fname} started, ncores={n_cores}, nchunks={len(iterable):,}...")

    if use_threads:
        with ThreadPool(
            processes=n_cores,
            initializer=initializer,
            initargs=initargs,
        ) as pool:
            res = tqdmu(pool.starmap(func, iterable))
    else:

        with Pool(
            processes=n_cores,
            initializer=initializer,
            initargs=initargs,
        ) as pool:
            res = tqdmu(pool.starmap(func, iterable))

    logger.info(f"Function applied.")

    if return_dataframe:
        res = pd.concat(res, ignore_index=True)
        logger.info(f"Results Concatenated.")
    return res


def set_tf_gpu(gpu: int):
    ensure_installed("tensorflow")

    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[3], "GPU")
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

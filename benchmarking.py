"""Live benchmarking."""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *
import numpy as np
from timeit import default_timer as timer

from pyutilz.system import tqdmu

# ----------------------------------------------------------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------------------------------------------------------


def benchmark_algos_by_runtime(implementations: list, algo_name: str = "", n_reps: int = 2, verbose: int = 0, *args, **kwargs) -> tuple:
    """For a list of implementations on the same algorithm (accepting same inputs),
    repeats them n_reps times and sorts them by min runtime.
    n_reps>=2 means prewarm is possible (good for jit-compiled functions)."""

    durations = []
    if verbose > 1:
        mes_parts = []
        logger.info(f"Benchmarking {algo_name} implementations...")

    for algo in tqdmu(implementations, desc=f"{algo_name} implementations", leave=False):
        min_duration = 1e20
        for rep_id in range(n_reps):
            algo_start = timer()
            _ = algo(*args, **kwargs)
            duration = timer() - algo_start
            if duration < min_duration:
                min_duration = duration
        durations.append(min_duration)
        if verbose > 1:
            mes_parts.append(f"{algo.__name__}: {duration:.3f} sec.")

    if verbose > 1:
        logger.info(",".join(mes_parts))

    durations = np.array(durations)
    indices = np.argsort(durations)

    sorted_implementations = [implementations[idx] for idx in indices]
    sorted_durations = durations[indices].tolist()

    return sorted_implementations, sorted_durations

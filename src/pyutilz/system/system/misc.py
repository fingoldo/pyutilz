#!$ pip install pympler psutil gpu-info pylspci gputil py-cpuinfo
#!$ pip install pycuda

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------


# ensure_installed("pympler psutil numba tqdm gpu-info")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import Any as _Any
from typing import Optional, Sequence, Union

import locale
import tqdm
import psutil
import tracemalloc
from pympler import asizeof
import platform, sys, importlib
from datetime import timezone, datetime

# ----------------------------------------------------------------------------------------------------------------------------
# IPython
# ----------------------------------------------------------------------------------------------------------------------------


def run_from_ipython():
    try:
        return bool(__IPYTHON__)  # type: ignore[name-defined]  # IPython-injected global, only exists inside an IPython/Jupyter session
    except NameError:
        return False


# ----------------------------------------------------------------------------------------------------------------------------
# Progressbars
# ----------------------------------------------------------------------------------------------------------------------------


def tqdmu(*args, **kwargs):

    if run_from_ipython():
        try:
            res = tqdm.tqdm_notebook(*args, **kwargs)
        except Exception:
            res = tqdm.tqdm(*args, **kwargs)
        return res
    else:
        return tqdm.tqdm(*args, **kwargs)


def tqdmu_lazy_start(iterable, *, min_total: int = 2, **kwargs):
    """Drop-in for ``tqdmu(iterable, **kwargs)`` that starts the elapsed
    timer at the FIRST iteration, not at bar construction.

    Motivation: the plain ``tqdmu(xs, desc=...)`` sets ``start_t`` at
    construction. If the caller interleaves heavy work (data conversion,
    I/O, another training phase) between building the iterable and the
    first pull, the bar displays that idle wall-clock as elapsed — e.g.
    a ``target_type: 0/1 [6:27:44<?]`` line sitting on the log for hours
    before the first real tick. The lazy variant defers ``reset()`` until
    the iterable actually yields its first item, so elapsed tracks real
    work only.

    Single-item loops (``min_total`` default 2): the bar is suppressed
    entirely when ``len(iterable) <= 1``. Reason — for an iterable of one,
    tqdm displays ``0/1 [HH:MM:SS<?]`` for the entire body of the first
    (and only) iteration, never reaching the post-yield ``update(1)``
    until the caller exits its outer loop. On long single-item loops
    this is more confusing than informative (looks like progress is
    stuck). Pass ``min_total=1`` to restore the old "always show a bar"
    behaviour.

    Minor trade-off: the bar is created with an empty underlying iterator
    and driven manually via ``.update(1)`` per yield, so indeterminate
    iterables lose their auto-total inference; pass ``total=...``
    explicitly in that case.
    """
    # Best-effort extract total so the progress percentage still works.
    if "total" not in kwargs:
        try:
            kwargs["total"] = len(iterable)
        except TypeError:
            pass

    # Suppress single-item bars when ``min_total>1`` — the caller doesn't
    # benefit from a 0/1 progress display that never updates until the
    # outer loop ends.
    _total = kwargs.get("total")
    if _total is not None and _total < min_total:
        for item in iterable:
            yield item
        return

    bar = tqdmu(iter([]), **kwargs)
    try:
        first = True
        for item in iterable:
            if first:
                bar.reset(total=kwargs.get("total"))
                first = False
            yield item
            bar.update(1)
    finally:
        bar.close()

# ----------------------------------------------------------------------------------------------------------------------------
# Software info
# ----------------------------------------------------------------------------------------------------------------------------


def get_os_info() -> dict:
    """Get detailed OS information.

    Returns:
        dict: OS system, machine, version, platform, architecture, edition (Windows)
    """
    info = {}

    current_system = platform.system()
    info["system"] = current_system
    info["machine"] = platform.machine()
    info["version"] = platform.version()
    info["platform"] = platform.platform()

    architecture: _Any = platform.architecture()
    if isinstance(architecture, tuple):
        architecture = ", ".join(architecture)

    info["architecture"] = architecture
    if current_system == "Windows":
        info["edition"] = platform.win32_edition()

    return info


def get_python_info() -> dict:
    """Get Python implementation and version details.

    Returns:
        dict: Python implementation, version, sys_version
    """
    res = {}
    res["implementation"] = platform.python_implementation()
    res["version"] = platform.python_version()
    res["sys_version"] = sys.version

    return res


def get_libs_versions(libs: Union[Sequence, str] = "numpy pandas numba", sep: str = " ") -> dict:

    if isinstance(libs, str):
        libs = libs.split(sep)

    res = {}

    for modulename in libs:
        try:
            importlib.import_module(modulename)
            version = sys.modules[modulename].__version__
            res[modulename] = version
        except Exception as e:  # nosec B110 - best-effort version probe across an arbitrary list of libs; missing/unimportable/version-less modules are expected and skipped, not a hidden bug
            logger.debug("Could not determine version for %s: %s", modulename, e)
    return res


def get_max_affordable_workers_count(reservedCores=1):
    import psutil

    n = psutil.cpu_count(logical=False) - reservedCores
    if n < 1:
        n = 1
    return n


def count_app_instances(processname=None, cmdline=None):
    import psutil

    n = 0
    # Iterate over all running process
    for proc in psutil.process_iter():
        if processname is not None:
            if processname != proc.name():
                continue
        if cmdline is not None:
            try:
                if cmdline not in proc.cmdline():
                    continue
            except Exception as e:  # nosec B112 - transient psutil access errors (process exited / no access) on a single proc while iterating all processes; skip that proc and keep counting
                logger.debug("Could not read cmdline for process %s: %s", proc, e)
                continue
        n = n + 1
    return n


def get_script_file(file: Optional[str] = __file__) -> str:
    import os

    return os.path.basename(file)


def report_large_objects(min_size_mb: int = 200, initial_memory_snapshot: object = None):

    report = "Large objects in RAM:"
    nbig = 0
    for name, obj in globals().items():
        if name != "asizeof":
            size_mb = round(asizeof.asizeof(obj) / 1024 / 1024, 1)
            if size_mb > min_size_mb:
                nbig += 1
                report = report + "\n" + f"{name}: {size_mb:,} MB".replace(",", "_")
    if nbig > 0:
        logger.info(report)
        if initial_memory_snapshot:
            try:
                current_snapshot = tracemalloc.take_snapshot()
                top_stats = current_snapshot.compare_to(initial_memory_snapshot, "lineno")
                print("Top stats:\n")
                print(top_stats)
            except Exception as e:
                logger.exception(e)

# ----------------------------------------------------------------------------------------------------------------------------
# Resource Monitoring
# ----------------------------------------------------------------------------------------------------------------------------


def ensure_idle_devices(
    duration_seconds: int = 5,
    max_cpu_load_percent: float = 10.0,
    min_cpu_free_ram_gb: float = 1.0,
    max_gpu_load_percent: float = 15.0,
    min_gpu_free_ram_gb: float = 1.0,
    gpu_ids: Optional[list] = None,
):
    """Ensure CPU and GPU devices are idle before running benchmarks.

    Monitors system resources and waits until all conditions are met for the specified
    duration. If conditions are violated, resets the timer and starts counting again.

    Args:
        duration_seconds: How long conditions must be met (seconds)
        max_cpu_load_percent: Maximum allowed CPU utilization (0-100%)
        min_cpu_free_ram_gb: Minimum required free CPU RAM (GB)
        max_gpu_load_percent: Maximum allowed GPU utilization (0-100%)
        min_gpu_free_ram_gb: Minimum required free GPU RAM (GB)
        gpu_ids: List of GPU IDs to monitor (empty = all GPUs)

    Returns:
        bool: True if conditions met, False if initial validation failed
    """
    if gpu_ids is None:
        gpu_ids = []
    import time

    if not (0 <= max_cpu_load_percent <= 100):
        raise ValueError("max_cpu_load_percent must be between 0 and 100")
    if not (0 <= max_gpu_load_percent <= 100):
        raise ValueError("max_gpu_load_percent must be between 0 and 100")

    logger.info("Ensuring idle devices for %s seconds with the following conditions:", duration_seconds)
    logger.info(
        f"  Max CPU load: {max_cpu_load_percent}%, Min CPU free RAM: {min_cpu_free_ram_gb} GB, "
        f"Max GPU load: {max_gpu_load_percent}%, Min GPU free RAM: {min_gpu_free_ram_gb} GB. "
        f"GPU IDs to check: {gpu_ids if gpu_ids else 'All'}."
    )

    def check_cpu_initial_conditions():
        total_cpu_ram_gb = psutil.virtual_memory().total / (1024**3)
        if min_cpu_free_ram_gb > total_cpu_ram_gb:
            logger.warning(f"Requested CPU free RAM ({min_cpu_free_ram_gb} GB) exceeds total available RAM ({total_cpu_ram_gb:.2f} GB)")
            return False
        return True

    def check_gpu_initial_conditions():
        try:
            import GPUtil
        except ImportError:
            logger.warning("GPUtil not available, skipping GPU checks")
            return True

        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            if gpu_ids and gpu.id not in gpu_ids:
                continue
            total_gpu_ram_gb = gpu.memoryTotal / 1024
            if min_gpu_free_ram_gb > total_gpu_ram_gb:
                logger.warning(f"Requested GPU {gpu.id} free RAM ({min_gpu_free_ram_gb} GB) exceeds total available RAM ({total_gpu_ram_gb:.2f} GB)")
                return False
        return True

    if not check_cpu_initial_conditions() or not check_gpu_initial_conditions():
        return False

    def check_cpu_conditions():
        cpu_load_percent = psutil.cpu_percent(percpu=False)
        cpu_free_ram_gb = psutil.virtual_memory().available / (1024**3)
        mes = ""
        if cpu_load_percent > max_cpu_load_percent:
            mes += f"CPU load too high: {cpu_load_percent:.2f}% > {max_cpu_load_percent:.2f}%"
        if cpu_free_ram_gb < min_cpu_free_ram_gb:
            if mes:
                mes += " "
            mes += f"CPU free RAM too low: {cpu_free_ram_gb:.2f} GB < {min_cpu_free_ram_gb:.2f} GB"
        if mes:
            logger.debug(mes)
        return cpu_load_percent <= max_cpu_load_percent and cpu_free_ram_gb >= min_cpu_free_ram_gb

    def check_gpu_conditions():
        try:
            import GPUtil
        except ImportError:
            return True  # Skip GPU checks if GPUtil not available

        gpus = GPUtil.getGPUs()
        all_conditions_met = True
        for gpu in gpus:
            if gpu_ids and gpu.id not in gpu_ids:
                continue
            gpu_load_percent = gpu.load * 100
            gpu_free_ram_gb = gpu.memoryFree / 1024
            mes = ""
            if gpu_load_percent > max_gpu_load_percent:
                mes += f"GPU {gpu.id} load too high: {gpu_load_percent:.2f}% > {max_gpu_load_percent:.2f}%"
                all_conditions_met = False
            if gpu_free_ram_gb < min_gpu_free_ram_gb:
                if mes:
                    mes += " "
                mes += f"GPU {gpu.id} free RAM too low: {gpu_free_ram_gb:.2f} GB < {min_gpu_free_ram_gb:.2f} GB"
                all_conditions_met = False
            if mes:
                logger.debug(mes)
        return all_conditions_met

    start_time = time.time()

    while True:
        if check_cpu_conditions() and check_gpu_conditions():
            elapsed_time = time.time() - start_time
            if elapsed_time >= duration_seconds:
                logger.info("Devices are idle and ready for benchmarking")
                return True
        else:
            start_time = time.time()

        time.sleep(1)

# ----------------------------------------------------------------------------------------------------------------------------
# Locales, Date/Time, etc.
# ----------------------------------------------------------------------------------------------------------------------------


def get_utc_unix_timestamp():

    return int(datetime.now(tz=timezone.utc).timestamp())


def get_locale_settings(locale_name: str = "", only_fields: Optional[tuple] = None) -> dict:
    """Return a dict of locale params.

    :param str locale_name: Desired locale name, or empty string for OS default locale.
    :param tuple only_fields: Desired locale fields, or None for all fields.

    >>>get_locale_settings(locale_name="en_US.utf8", only_fields=("decimal_point", "thousands_sep"))
    {'decimal_point': '.', 'thousands_sep': ','}

    """
    locale.setlocale(locale.LC_ALL, locale_name)
    settings: _Any = locale.localeconv()
    if settings:
        if only_fields:
            settings = {field: value for field, value in settings.items() if field in only_fields}
    return settings

# ----------------------------------------------------------------------------------------------------------------------------
# Sounds
# ----------------------------------------------------------------------------------------------------------------------------


def beep():
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    try:
        import winsound

        winsound.Beep(frequency, duration)
    except Exception as e:  # nosec B110 - best-effort audible notification; winsound is Windows-only/optional and beep failure must never break the caller
        logger.debug("Could not play beep sound: %s", e)

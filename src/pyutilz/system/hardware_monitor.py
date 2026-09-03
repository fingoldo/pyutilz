"""Background CPU/RAM (and optionally GPU) utilization monitor for profiling long-running jobs."""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

import psutil
import threading
import numpy as np

from pyutilz.system.system import get_own_memory_usage, get_nvidia_smi_info
from pyutilz.system.psutil_compat import get_cpu_freq_current_mhz, missing_psutil_functions
from pyutilz.core.pythonlib import to_float
from typing import Any as _Any, List as _List, Optional

# ----------------------------------------------------------------------------------------------------------------------------
# Hardware Utilization Monitor
# ----------------------------------------------------------------------------------------------------------------------------


class UtilizationMonitor:
    """Background thread monitor for CPU/GPU/RAM utilization.

    Continuously monitors hardware utilization in a background thread:
    - CPU: load %, clock frequency
    - RAM: total used/free, own process usage
    - GPU: load %, VRAM used/free, temperature, power draw, clock frequency

    Example:
        >>> monitor = UtilizationMonitor(gpu_ids=[0])
        >>> monitor.start()
        >>> # ... run your code ...
        >>> monitor.stop()
        >>> avg_util = monitor.get_average_utilization()
        >>> print(f"Average CPU load: {avg_util['cpu_utilizaton_percent']}%")
    """

    def __init__(
        self,
        sleep_interval_seconds: float = 1.0,
        gpu_ids: Optional[list] = None,
    ):
        """Initialize the utilization monitor.

        Args:
            sleep_interval_seconds: Sampling interval (default 1.0 second)
            gpu_ids: List of GPU IDs to monitor (empty = all GPUs)
        """
        if gpu_ids is None:
            gpu_ids = []
        self.cpu_utilizaton: _List[_Any] = []
        self.cpu_clocks: _List[_Any] = []

        self.own_ram_used: _List[_Any] = []
        self.total_ram_used: _List[_Any] = []
        self.total_ram_free: _List[_Any] = []

        self.mean_gpu_ram_free: _List[_Any] = []
        self.mean_gpu_ram_used: _List[_Any] = []

        self.mean_gpu_clocks: _List[_Any] = []
        self.mean_gpu_utilizaton: _List[_Any] = []

        self.mean_gpu_power_draw: _List[_Any] = []
        self.mean_gpu_temp: _List[_Any] = []

        self.gpu_ids = gpu_ids
        self.sleep_interval_seconds = sleep_interval_seconds
        self.n_samples = 0
        self.n_sampling_errors = 0
        self.stop_flag = threading.Event()
        # daemon=True (regression fix): if the code between start()/stop() raises -- the entire
        # reason someone wraps a benchmark in a monitor, to profile code that might fail --
        # stop() is skipped and stop_flag is never set. A non-daemon thread then runs its
        # infinite loop forever, and CPython refuses to exit the interpreter until every
        # non-daemon thread finishes -- not just "a thread leaks", the whole process hangs at
        # shutdown. A daemon thread no longer blocks interpreter exit either way.
        self.thread = threading.Thread(target=self.query_utilization, daemon=True)

    def __getstate__(self) -> dict:
        """Drop the unpicklable ``threading.Event``/``threading.Thread`` (fresh ones are created in
        ``__setstate__``) -- flagged by ``pyutilz.dev.code_audit.unpicklable_resource_state``; this
        monitor isn't on any current pickling path, but the guard is cheap and matches the repo-wide
        convention. The restored instance is NOT running -- call ``start()`` again if needed."""
        state = self.__dict__.copy()
        state["stop_flag"] = None
        state["thread"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state and re-create the ``Event``/``Thread`` dropped by ``__getstate__``, in
        the same stopped (not-yet-started) state a fresh ``__init__`` would leave them in."""
        self.__dict__.update(state)
        self.stop_flag = threading.Event()
        self.thread = threading.Thread(target=self.query_utilization, daemon=True)

    def query_utilization(self):
        """Background thread function that monitors hardware utilization.

        Monitors CPU load %, clocks, RAM free/used.
        Also monitors per-GPU: load %, RAM free/used, temp, clocks, power draw.
        """
        while not self.stop_flag.is_set():
            try:
                self._collect_sample()
            except Exception:
                # A single bad sample (nvidia-smi reporting "N/A" for power_draw, a malformed XML
                # payload) used to kill the thread outright, after which stop() still returned
                # cleanly and the truncated series was presented as a complete run profile.
                self.n_sampling_errors += 1
                logger.exception("Hardware utilization sampling failed; continuing with the next sample")
            # Waiting on the stop flag (rather than sleeping) lets stop() return immediately
            # instead of blocking for up to a full interval, and sampling BEFORE the wait means a
            # run shorter than one interval still yields data.
            self.stop_flag.wait(self.sleep_interval_seconds)

    def _collect_sample(self) -> None:
        """Take one CPU/RAM/GPU sample and append it to the accumulators."""
        # CPU
        self.cpu_utilizaton.append(psutil.cpu_percent(percpu=False))
        # CPU frequency is optional twice over: psutil has no cpu_freq attribute at all on some
        # platforms (macOS), and where it exists it can still return None on VMs/containers that
        # do not expose the counter. Both are handled absences, not sampling errors -- and neither
        # is recorded as 0.0, which would be averaged in later as a real "CPU ran at 0 MHz"
        # reading. A sample with no clock simply contributes no clock, and get_average_utilization
        # reports cpu_clocks_mhz=None plus the unavailable-metric list.
        cpu_clock_mhz = get_cpu_freq_current_mhz(psutil_module=psutil)
        if cpu_clock_mhz is not None:
            self.cpu_clocks.append(cpu_clock_mhz)

        # RAM
        # get_own_memory_usage() documents returning None if psutil raises AND there's no
        # prior successful reading to fall back on (e.g. psutil fails on the very first
        # sample) -- appending it unfiltered would make every later np.mean() over this list
        # raise TypeError, so skip the sample instead of recording a None.
        own_ram = get_own_memory_usage()
        if own_ram is not None:
            self.own_ram_used.append(own_ram)
        mem = psutil.virtual_memory()
        self.total_ram_used.append(mem.used)
        self.total_ram_free.append(mem.free)

        # GPU
        gpu_stats = get_nvidia_smi_info(include_stats=True)

        if gpu_stats is None:
            self.n_samples += 1
            return

        total_gpu_ram_free = 0.0
        total_gpu_ram_used = 0.0

        total_gpu_clocks = 0.0
        total_gpu_utilizaton = 0.0

        total_gpu_power_draw = 0.0
        total_gpu_temp = 0.0

        n = 0

        for gpu_info in gpu_stats.get("gpu", []):
            # gpu_module_id may be a non-numeric string (e.g. from malformed
            # nvidia-smi output); coerce safely instead of raising ValueError.
            raw_gpu_id = gpu_info.get("gpu_module_id", 0)
            try:
                gpu_id = int(raw_gpu_id)
            except (TypeError, ValueError):
                gpu_id = 0
            if self.gpu_ids and gpu_id not in self.gpu_ids:
                continue

            fb_memory_usage = gpu_info.get("fb_memory_usage", {})
            free_mem = fb_memory_usage.get("free", "0 MiB")
            used_mem = fb_memory_usage.get("used", "0 MiB")
            total_gpu_ram_free += to_float(str(free_mem).replace(" MiB", ""))
            total_gpu_ram_used += to_float(str(used_mem).replace(" MiB", ""))

            utilization = gpu_info.get("utilization", {})
            gpu_util = utilization.get("gpu_util", "0 %")
            total_gpu_utilizaton += to_float(str(gpu_util).replace(" %", ""))

            temperature = gpu_info.get("temperature", {})
            gpu_temp = temperature.get("gpu_temp", "0 C")
            total_gpu_temp += to_float(str(gpu_temp).replace(" C", ""))

            power_readings = gpu_info.get("power_readings", {})
            power_draw = power_readings.get("power_draw", "0 W")
            total_gpu_power_draw += to_float(str(power_draw).replace(" W", ""))

            clocks = gpu_info.get("clocks", {})
            sm_clock = clocks.get("sm_clock", "0 MHz")
            total_gpu_clocks += to_float(str(sm_clock).replace(" MHz", ""))

            n += 1

        if n:
            self.mean_gpu_ram_free.append(total_gpu_ram_free / n)
            self.mean_gpu_ram_used.append(total_gpu_ram_used / n)

            self.mean_gpu_clocks.append(total_gpu_clocks / n)
            self.mean_gpu_utilizaton.append(total_gpu_utilizaton / n)

            self.mean_gpu_power_draw.append(total_gpu_power_draw / n)
            self.mean_gpu_temp.append(total_gpu_temp / n)

        # Counted at the END, not before the parse loop above: the loop can raise (a card
        # reporting "N/A"), and the caller then increments n_sampling_errors for the SAME sample --
        # 100 consecutively-failing samples used to be reported as "100 of 200 failed".
        self.n_samples += 1

    def start(self) -> None:
        """Start the background monitoring thread.

        The Thread object is (re)created here rather than in ``__init__``, so the same instance
        can be started, stopped and started again -- including reuse as a context manager, which
        otherwise raised "threads can only be started once" on the second ``with``.
        """
        if self.thread is not None and self.thread.is_alive():
            logger.debug("Hardware utilization monitoring already running")
            return
        self.stop_flag.clear()
        self.thread = threading.Thread(target=self.query_utilization, daemon=True)
        self.thread.start()
        logger.info("Hardware utilization monitoring started")

    def stop(self, timeout: Optional[float] = None) -> None:
        """Stop the background monitoring thread and wait for it to finish."""
        self.stop_flag.set()
        # ``is_alive()``, not just ``is not None``: __init__ already assigns an UNSTARTED Thread, so
        # stop() on a never-started monitor raised "cannot join thread before it is started" --
        # which, from a try/finally, masked the caller's real exception.
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout)
        logger.info("Hardware utilization monitoring stopped")

    def __enter__(self) -> "UtilizationMonitor":
        """``with UtilizationMonitor(...) as monitor:`` -- guarantees stop() runs via __exit__
        even when the wrapped code raises, so a benchmark exception never skips cleanup."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def get_average_utilization(self, ndigits: int = 3):
        """Calculate average utilization across all samples.

        Args:
            ndigits: Number of decimal places for rounding

        Returns:
            dict: Average utilization metrics for CPU, RAM, and GPU
        """
        if self.n_sampling_errors:
            logger.warning(
                "get_average_utilization: %d of %d samples failed; the averages below cover only the successful ones",
                self.n_sampling_errors, self.n_samples + self.n_sampling_errors,
            )
        if self.cpu_utilizaton:
            return dict(
                n_samples=len(self.cpu_utilizaton),
                n_sampling_errors=self.n_sampling_errors,
                unavailable_metrics=missing_psutil_functions(psutil),
                cpu_utilizaton_percent=round(np.mean(self.cpu_utilizaton), ndigits),
                cpu_clocks_mhz=round(np.mean(self.cpu_clocks), ndigits) if self.cpu_clocks else None,
                # ``if <series> else None``, like every sibling field below: np.mean([]) is nan,
                # and a consumer skipping None-valued metrics silently averaged that nan in.
                own_ram_used_gb=round(np.mean(self.own_ram_used), ndigits) if self.own_ram_used else None,
                total_ram_used_gb=round(np.mean(self.total_ram_used) / 1024**3, ndigits) if self.total_ram_used else None,
                total_ram_free_gb=round(np.mean(self.total_ram_free) / 1024**3, ndigits) if self.total_ram_free else None,
                gpu_ram_free_gb=round(np.mean(self.mean_gpu_ram_free) / 1024, ndigits) if self.mean_gpu_ram_free else None,
                gpu_ram_used_gb=round(np.mean(self.mean_gpu_ram_used) / 1024, ndigits) if self.mean_gpu_ram_used else None,
                gpu_clocks_mhz=round(np.mean(self.mean_gpu_clocks), ndigits) if self.mean_gpu_clocks else None,
                gpu_utilizaton_percent=round(np.mean(self.mean_gpu_utilizaton), ndigits) if self.mean_gpu_utilizaton else None,
                gpu_power_draw_watt=round(np.mean(self.mean_gpu_power_draw), ndigits) if self.mean_gpu_power_draw else None,
                gpu_temp_celsius=round(np.mean(self.mean_gpu_temp), ndigits) if self.mean_gpu_temp else None,
            )
        else:
            return dict(
                n_samples=0,
                n_sampling_errors=self.n_sampling_errors,
                unavailable_metrics=missing_psutil_functions(psutil),
                cpu_utilizaton_percent=None,
                cpu_clocks_mhz=None,
                own_ram_used_gb=None,
                total_ram_used_gb=None,
                total_ram_free_gb=None,
                gpu_ram_free_gb=None,
                gpu_ram_used_gb=None,
                gpu_clocks_mhz=None,
                gpu_utilizaton_percent=None,
                gpu_power_draw_watt=None,
                gpu_temp_celsius=None,
            )

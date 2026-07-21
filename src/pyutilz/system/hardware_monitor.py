"""Background CPU/RAM (and optionally GPU) utilization monitor for profiling long-running jobs."""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

import time
import psutil
import threading
import numpy as np

from pyutilz.system.system import get_own_memory_usage, get_nvidia_smi_info
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
        self.stop_flag = threading.Event()
        self.thread = threading.Thread(target=self.query_utilization)

    def query_utilization(self):
        """Background thread function that monitors hardware utilization.

        Monitors CPU load %, clocks, RAM free/used.
        Also monitors per-GPU: load %, RAM free/used, temp, clocks, power draw.
        """
        while not self.stop_flag.is_set():
            time.sleep(self.sleep_interval_seconds)

            # CPU
            self.cpu_utilizaton.append(psutil.cpu_percent(percpu=False))
            # cpu_freq() can return None on platforms/VMs where the frequency
            # is unavailable; fall back to 0.0 rather than crashing the thread.
            cpu_freq = psutil.cpu_freq(percpu=False)
            self.cpu_clocks.append(cpu_freq.current if cpu_freq is not None else 0.0)

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
                continue

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

    def start(self):
        """Start the background monitoring thread."""
        self.thread.start()
        logger.info("Hardware utilization monitoring started")

    def stop(self):
        """Stop the background monitoring thread and wait for it to finish."""
        self.stop_flag.set()
        self.thread.join()
        logger.info("Hardware utilization monitoring stopped")

    def get_average_utilization(self, ndigits: int = 3):
        """Calculate average utilization across all samples.

        Args:
            ndigits: Number of decimal places for rounding

        Returns:
            dict: Average utilization metrics for CPU, RAM, and GPU
        """
        if self.cpu_utilizaton:
            return dict(
                cpu_utilizaton_percent=round(np.mean(self.cpu_utilizaton), ndigits),
                cpu_clocks_mhz=round(np.mean(self.cpu_clocks), ndigits),
                own_ram_used_gb=round(np.mean(self.own_ram_used), ndigits),
                total_ram_used_gb=round(np.mean(self.total_ram_used) / 1024**3, ndigits),
                total_ram_free_gb=round(np.mean(self.total_ram_free) / 1024**3, ndigits),
                gpu_ram_free_gb=round(np.mean(self.mean_gpu_ram_free) / 1024, ndigits) if self.mean_gpu_ram_free else None,
                gpu_ram_used_gb=round(np.mean(self.mean_gpu_ram_used) / 1024, ndigits) if self.mean_gpu_ram_used else None,
                gpu_clocks_mhz=round(np.mean(self.mean_gpu_clocks), ndigits) if self.mean_gpu_clocks else None,
                gpu_utilizaton_percent=round(np.mean(self.mean_gpu_utilizaton), ndigits) if self.mean_gpu_utilizaton else None,
                gpu_power_draw_watt=round(np.mean(self.mean_gpu_power_draw), ndigits) if self.mean_gpu_power_draw else None,
                gpu_temp_celsius=round(np.mean(self.mean_gpu_temp), ndigits) if self.mean_gpu_temp else None,
            )
        else:
            return dict(
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

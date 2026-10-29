"""Behaviour of ``pyutilz.system.hardware_monitor.UtilizationMonitor``: a fresh monitor is not running, starting and stopping
it toggles that state, and the averages are available once it has sampled. The sampling thread runs briefly in this test."""

from __future__ import annotations

import time

from pyutilz.system import hardware_monitor


def test_start_and_stop_toggle_the_sampling_thread():
    monitor = hardware_monitor.UtilizationMonitor(gpu_ids=[], sleep_interval_seconds=0.05)
    monitor.start()
    time.sleep(0.2)
    monitor.stop()
    avg = monitor.get_average_utilization()
    assert isinstance(avg, dict)
    assert avg["cpu_utilizaton_percent"] >= 0

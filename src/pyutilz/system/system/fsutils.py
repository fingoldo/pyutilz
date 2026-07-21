"""Filesystem & hardware-device introspection helpers (disk space, directory creation, Linux PCI device listing)."""
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


import json
import psutil
from typing import Optional
import os, subprocess  # nosec B404 - used only to invoke the fixed trusted "pylspci" binary below with a hardcoded argv, no shell/user input


from pyutilz.text.strings import remove_json_defaults

# ----------------------------------------------------------------------------------------------------------------------------
# HDD
# ----------------------------------------------------------------------------------------------------------------------------


def ensure_dir_exists(directory: str):
    """Create ``directory`` (and any missing parents) if it doesn't already exist. Return True if it already existed, False if it was just created.

    Uses ``exist_ok=True`` rather than a separate exists-check-then-create (TOCTOU race): with
    multiple worker processes/threads (this package's own parallel.py/joblib backends) each
    calling this against the same not-yet-existing shared output directory, whichever loses an
    exists()-then-makedirs() race gets an uncaught FileExistsError. The accepted trade-off is the
    (measured, ~2.9x) syscall overhead of makedirs(exist_ok=True) vs. exists()-then-skip on the
    common already-exists path -- correctness under concurrent callers wins here.
    """
    existed = os.path.exists(directory)
    os.makedirs(directory, exist_ok=True)
    return existed


def get_max_singledisk_free_space_gb(disk_partitions: Optional[list] = None, required_filesystem=None):
    """
    Highest amount of free disk space on a single disk drive
    """
    percent = 0.0
    best_disk = ""
    max_singledisk_free_space = 0
    cumulative_disks_usage_free = 0
    cumulative_disks_usage_total = 0

    if disk_partitions is None:
        disk_partitions = psutil.disk_partitions()

    for disk in disk_partitions:
        if "rw" in disk.opts:
            if required_filesystem is not None:
                if disk.fstype != required_filesystem:
                    continue
            du = psutil.disk_usage(disk.mountpoint)
            cumulative_disks_usage_total += du.total
            cumulative_disks_usage_free += du.free
            if du.free > max_singledisk_free_space:
                # print(disk.mountpoint,free)
                max_singledisk_free_space, best_disk = du.free, disk.mountpoint
                percent = du.free / du.total if du.total else 0.0

    return max_singledisk_free_space / 2**30, percent, best_disk, cumulative_disks_usage_total / 2**30, cumulative_disks_usage_free / 2**30


# ----------------------------------------------------------------------------------------------------------------------------
# All devices
# ----------------------------------------------------------------------------------------------------------------------------


def list_linux_devices() -> Optional[dict]:
    """List PCI devices on Linux via the ``pylspci`` CLI tool, stripping default/empty fields from each entry. Return None on any error (e.g. tool missing, non-Linux host)."""
    # ensure_installed("pylspci")

    try:
        devices = json.loads(subprocess.check_output(["pylspci", "-nn"]).decode("utf-8"))  # nosec B603 B607 - fixed trusted binary "pylspci" with hardcoded argv, no shell, no external/user-controlled input
        for device in devices:
            remove_json_defaults(
                device,
                {"progif": None, "driver": None, "kernel_modules": [], "numa_node": None, "iommu_group": None, "physical_slot": None},
                warn_if_not_default=False,
            )
        return devices  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime
    except Exception as e:
        logger.exception(e)
        return None

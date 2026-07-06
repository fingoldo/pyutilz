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
import os, subprocess  # nosec B404 - used only to invoke the fixed trusted "pylspci" binary below with a hardcoded argv, no shell/user input


from pyutilz.text.strings import remove_json_defaults

# ----------------------------------------------------------------------------------------------------------------------------
# HDD
# ----------------------------------------------------------------------------------------------------------------------------


def ensure_dir_exists(directory: str):
    # directory = os.path.dirname(file_path)
    existed = os.path.exists(directory)
    if not existed:
        os.makedirs(directory)
    return existed


def get_max_singledisk_free_space_gb(disk_partitions: object = None, required_filesystem=None):
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
                percent = du.free / du.total

    return max_singledisk_free_space / 2**30, percent, best_disk, cumulative_disks_usage_total / 2**30, cumulative_disks_usage_free / 2**30


# ----------------------------------------------------------------------------------------------------------------------------
# All devices
# ----------------------------------------------------------------------------------------------------------------------------


def list_linux_devices() -> dict:
    # ensure_installed("pylspci")

    try:
        devices = json.loads(subprocess.check_output(["pylspci", "-nn"]).decode("utf-8"))  # nosec B603 B607 - fixed trusted binary "pylspci" with hardcoded argv, no shell, no external/user-controlled input
        for device in devices:
            remove_json_defaults(
                device,
                {"progif": None, "driver": None, "kernel_modules": [], "numa_node": None, "iommu_group": None, "physical_slot": None},
                warn_if_not_default=False,
            )
        return devices
    except Exception as e:
        logger.exception(e)
        pass

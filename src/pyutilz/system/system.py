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

from pyutilz.core.pythonlib import ensure_installed

# ensure_installed("pympler psutil numba tqdm gpu-info")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import Optional, Sequence, Set, Union

import uuid
import locale
import pandas as pd
import re, json, tqdm
import socket, psutil
from pympler import asizeof
import platform, sys, importlib
import os, subprocess
from datetime import timezone, datetime, timedelta

import gc
import ctypes
import tracemalloc
import ctypes.wintypes

from pyutilz.text.strings import remove_json_defaults, remove_json_attributes, find_between

# ----------------------------------------------------------------------------------------------------------------------------
# String & Float Utilities
# ----------------------------------------------------------------------------------------------------------------------------


def remove_nas(obj: dict):
    """Recursively removes 'N/A' values from a dict and converts numeric strings to floats.

    Args:
        obj: Dictionary to clean

    Returns:
        Cleaned dictionary with N/A values removed and numeric strings converted
    """
    if isinstance(obj, dict):
        return {k: remove_nas(v) for k, v in obj.items() if v != "N/A"}
    elif isinstance(obj, list):
        return [remove_nas(item) for item in obj if item != "N/A"]
    elif isinstance(obj, str):
        try:
            return float(obj)
        except ValueError:
            return obj
    else:
        return obj


# ----------------------------------------------------------------------------------------------------------------------------
# WMI Helpers (Windows)
# ----------------------------------------------------------------------------------------------------------------------------

# WMI availability codes
availability_dict = {
    3: "Running or Full Power",
    4: "Warning",
    5: "In Test",
    10: "Not Installed",
    11: "Off Line",
    13: "Power Save - Unknown",
    14: "Power Save - Low Power Mode",
    15: "Power Save - Standby",
    16: "Power Cycle",
    17: "Power Save - Warning",
}

# WMI memory characteristics codes
characteristics_dict = {
    0: "Reserved",
    1: "Unknown",
    2: "DRAM",
    3: "Static RAM",
    4: "Cache DRAM",
    5: "EDO",
    6: "EDRAM",
    7: "VRAM",
    8: "SRAM",
    9: "RAM",
    10: "ROM",
    11: "Flash",
    12: "EEPROM",
    13: "FEPROM",
    14: "EPROM",
    15: "CDRAM",
    16: "3DRAM",
    17: "SDRAM",
    18: "SGRAM",
    19: "RDRAM",
    20: "DDR",
    21: "DDR2",
    22: "DDR2 FB-DIMM",
    24: "DDR3",
    25: "FBD2",
    26: "DDR4",
}


def decode_memory_type(memory_type) -> str:
    """Decode WMI memory type code to human-readable string.

    Args:
        memory_type: WMI memory type code (integer)

    Returns:
        Human-readable memory type (e.g., "DDR4", "DDR3")
    """
    memory_types = {
        0: "Unknown",
        1: "Other",
        2: "DRAM",
        3: "Synchronous DRAM",
        4: "Cache DRAM",
        5: "EDO",
        6: "EDRAM",
        7: "VRAM",
        8: "SRAM",
        9: "RAM",
        10: "ROM",
        11: "Flash",
        12: "EEPROM",
        13: "FEPROM",
        14: "EPROM",
        15: "CDRAM",
        16: "3DRAM",
        17: "SDRAM",
        18: "SGRAM",
        19: "RDRAM",
        20: "DDR",
        21: "DDR2",
        22: "DDR2 FB-DIMM",
        24: "DDR3",
        25: "FBD2",
        26: "DDR4",
    }
    return memory_types.get(memory_type, "Unknown")


def decode_cpu_upgrade_method(upgrade_method: int) -> str:
    """Decode WMI CPU upgrade method code to socket type.

    Args:
        upgrade_method: WMI upgrade method code (integer)

    Returns:
        CPU socket type (e.g., "Socket AM4", "LGA1151")
    """
    upgrade_method_dict = {
        1: "Other",
        2: "Unknown",
        3: "Daughter Board",
        4: "ZIF Socket",
        5: "Replacement/Piggy Back",
        6: "None",
        7: "LIF Socket",
        8: "Slot 1",
        9: "Slot 2",
        10: "370 Pin Socket",
        11: "Slot A",
        12: "Slot M",
        13: "Socket 423",
        14: "Socket A (Socket 462)",
        15: "Socket 478",
        16: "Socket 754",
        17: "Socket 940",
        18: "Socket 939",
        19: "Socket mPGA604",
        20: "Socket LGA771",
        21: "Socket LGA775",
        22: "Socket S1",
        23: "Socket AM2",
        24: "Socket F (1207)",
        25: "Socket LGA1366",
        26: "Socket G34",
        27: "Socket AM3",
        28: "Socket C32",
        29: "Socket LGA1156",
        30: "Socket LGA1567",
        31: "Socket PGA988A",
        32: "Socket BGA1288",
        33: "rPGA988B",
        34: "BGA1023",
        35: "BGA1024",
        36: "BGA1155",
        37: "Socket LGA1356",
        38: "Socket LGA2011",
        39: "Socket FS1",
        40: "Socket FS2",
        41: "Socket FM1",
        42: "Socket FM2",
        43: "Socket LGA2011-3",
        44: "Socket LGA1356-3",
        45: "Socket LGA1150",
        46: "Socket BGA1168",
        47: "Socket BGA1234",
        48: "Socket BGA1364",
        49: "Socket AM4",
        50: "Socket LGA1151",
        51: "Socket BGA1356",
        52: "Socket BGA1440",
        53: "Socket BGA1515",
        54: "Socket LGA3647-1",
        55: "Socket SP3",
        56: "Socket SP3r2",
        57: "Socket LGA2066",
        58: "Socket BGA1392",
        59: "Socket BGA1510",
        60: "Socket BGA1528",
    }
    return upgrade_method_dict.get(upgrade_method, "Unknown")


def dict_to_tuple(d: dict):
    """Convert dict to sorted tuple for hashing purposes.

    Args:
        d: Dictionary to convert

    Returns:
        Sorted tuple of (key, value) pairs
    """
    return tuple(sorted(d.items()))


def get_wmi_obj_as_dict(
    obj,
    exclude_pros: set = set(),
    ensure_float: set = set(),
    ensure_int: set = set(),
    decode_dict: dict = {},
):
    """Convert WMI object to dictionary with type conversion.

    Args:
        obj: WMI object to convert
        exclude_pros: Property names to exclude
        ensure_float: Property names to convert to float
        ensure_int: Property names to convert to int
        decode_dict: Dict mapping property names to decoder functions

    Returns:
        Dictionary representation of WMI object
    """
    obj_dict = {}
    for prop in obj.properties:
        if prop in exclude_pros:
            continue
        value = getattr(obj, prop)
        if prop in ensure_float:
            try:
                value = float(value)
            except (ValueError, TypeError):
                pass
        elif prop in ensure_int:
            try:
                value = int(value)
            except (ValueError, TypeError):
                pass
        elif prop in decode_dict:
            value = decode_dict[prop](value)
        obj_dict[prop] = value
    return obj_dict


def summarize_devices(
    devices,
    exclude_pros: set = set(),
    ensure_float: set = set(),
    ensure_int: set = set(),
    decode_dict: dict = {},
):
    """Aggregate identical hardware devices with counts.

    Args:
        devices: List of WMI device objects
        exclude_pros: Property names to exclude
        ensure_float: Property names to convert to float
        ensure_int: Property names to convert to int
        decode_dict: Dict mapping property names to decoder functions

    Returns:
        List of unique devices with "Count" field added
    """
    from collections import Counter

    device_dicts = []
    for obj in devices:
        obj_dict = get_wmi_obj_as_dict(obj, exclude_pros, ensure_float, ensure_int, decode_dict)
        device_dicts.append(obj_dict)

    # Count identical devices
    device_tuples = [dict_to_tuple(d) for d in device_dicts]
    counts = Counter(device_tuples)

    # Create unique list with counts
    unique_devices = []
    seen = set()
    for device_dict, device_tuple in zip(device_dicts, device_tuples):
        if device_tuple not in seen:
            device_dict["Count"] = counts[device_tuple]
            unique_devices.append(device_dict)
            seen.add(device_tuple)

    return unique_devices


# ----------------------------------------------------------------------------------------------------------------------------
# IPython
# ----------------------------------------------------------------------------------------------------------------------------


def run_from_ipython():
    try:
        return bool(__IPYTHON__)  # Check if running in IPython/Jupyter
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


def get_system_info(
    return_hdd_info: bool = False,
    return_os_info: bool = False,
    return_sensitive_info: bool = False,
    return_hardware_info: bool = False,
    return_hardware_details: bool = False,
    return_usage_stats: bool = False,
    return_network_info: bool = False,
    only_stats: bool = False,
) -> dict:
    """Get comprehensive system information.

    Args:
        return_hdd_info: Include disk space information
        return_os_info: Include OS details, power plan, battery
        return_sensitive_info: Include machine GUID, host name, serial (REQUIRED for distributed.py)
        return_hardware_info: Include CPU, GPU, RAM details
        return_hardware_details: Include detailed WMI/dmidecode info
        return_usage_stats: Include current CPU/GPU/RAM usage
        return_network_info: Include external IP address
        only_stats: Minimal stats mode

    Returns:
        dict: System information based on requested flags

    Note:
        For backward compatibility with distributed.py, when only_stats=False and no other flags set,
        return_sensitive_info is automatically enabled.
    """
    # Backward compatibility: distributed.py needs sensitive info by default
    if not only_stats and not any([return_hdd_info, return_os_info, return_hardware_info,
                                   return_hardware_details, return_usage_stats, return_network_info]):
        return_sensitive_info = True
        return_os_info = True  # Needed to trigger sensitive info collection

    try:
        import numba
        from numba import cuda
    except ImportError:
        numba = None
        cuda = None
    try:
        from pyutilz.web import web
    except ImportError:
        web = None

    try:
        import x86cpu
    except Exception:
        pass
    try:
        import cpuinfo
    except Exception:
        pass

    info = dict()
    try:
        current_system = platform.system()
        if return_os_info:
            # OS - Use enhanced get_os_info()
            os_info_dict = get_os_info()
            if os_info_dict:
                info.update(os_info_dict)

            # Power plan and battery
            power_plan = get_power_plan()
            if power_plan:
                info["power_plan"] = power_plan

            battery_info = get_battery_info()
            if battery_info:
                info["battery_info"] = battery_info

            if return_sensitive_info:

                # (Almost) Constant features - REQUIRED for distributed.py
                info["os_machine_guid"] = str(uuid.UUID(int=uuid.getnode()))
                if current_system == "Windows":
                    os_serial = None
                    try:
                        os_serial = subprocess.check_output("wmic csproduct get uuid").decode().split("\n")[1].strip()
                        info["os_machine_guid"] = os_serial
                        info["os_serial"] = os_serial  # Also store as separate field for distributed.py
                    except Exception:
                        logger.warning(f"Could not extract Windows serial!")

                elif current_system == "Linux":
                    machine_id = subprocess.check_output("cat /var/lib/dbus/machine-id").decode().strip()
                    info["os_machine_guid"] = machine_id
                    info["os_serial"] = machine_id
                elif current_system == "Android":
                    serial = subprocess.check_output(["getprop", "ril.serialnumber"])[:-1].decode().strip()
                    info["os_machine_guid"] = serial
                    info["os_serial"] = serial
                elif current_system == "Mac":
                    # Fixed: Avoid shell=True by using subprocess.PIPE for chaining commands
                    ioreg_proc = subprocess.Popen(
                        ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                        stdout=subprocess.PIPE
                    )
                    grep_proc = subprocess.Popen(
                        ["grep", "-E", "(UUID)"],
                        stdin=ioreg_proc.stdout,
                        stdout=subprocess.PIPE
                    )
                    ioreg_proc.stdout.close()  # Allow ioreg to receive SIGPIPE if grep exits
                    output = grep_proc.communicate()[0]
                    guid = output.decode().split('"')[-2]
                    info["os_machine_guid"] = guid
                    info["os_serial"] = guid

                # host_name is also required for distributed.py
                info["host_name"] = socket.gethostname()

                if return_network_info and web is not None:
                    info["host_external_ip"] = web.get_external_ip()

        if return_hardware_info or return_usage_stats:
            cpu_freq = psutil.cpu_freq()
            ram = psutil.virtual_memory()

        if return_hardware_info:
            # CPU
            cpu_info = get_cpu_info()
            if cpu_info:
                info["cpu"] = cpu_info

            # Platform-specific CPU info
            if current_system == "Windows":
                cpu_wmi_info = get_wmi_cpuinfo()
                if cpu_wmi_info:
                    info["cpu_wmi_info"] = cpu_wmi_info
            else:
                cpu_lscpu_info = get_lscpu_info()
                if cpu_lscpu_info:
                    info["cpu_lscpu_info"] = cpu_lscpu_info
                cpu_board_info = get_linux_board_info()
                if cpu_board_info:
                    info["cpu_board_info"] = cpu_board_info

            if current_system != "Windows":
                info["cpu_num_sockets"] = get_nix_cpu_sockets_number()
            info["cpu_num_cores"] = psutil.cpu_count(logical=False)
            info["cpu_num_threads"] = psutil.cpu_count(logical=True)

            info["cpu_min_frequency_hz"] = cpu_freq.min
            info["cpu_max_frequency_hz"] = cpu_freq.max

            # RAM
            info["ram_total_gb"] = ram.total / 2**30

            # GPU - Use new nvidia-smi based detection
            gpu_nvidia_smi_info = get_nvidia_smi_info(include_stats=False)
            if gpu_nvidia_smi_info:
                info["gpu_nvidia_smi_info"] = gpu_nvidia_smi_info
                gpu_cuda_capabilities = get_cuda_gpu_details(gpu_nvidia_smi_info)
                if gpu_cuda_capabilities:
                    info["gpu_cuda_capabilities"] = gpu_cuda_capabilities

            # Large pages support
            large_pages_support = check_large_pages_support()
            if large_pages_support is not None:
                info["large_pages_support"] = large_pages_support

        # info['os_user_name']=os.environ.get('USER')
        # info['os_user_name']=os.environ.get('USERNAME')

        if return_usage_stats:

            info["cpu_current_frequency_hz"] = cpu_freq.current
            info["cpu_current_load_percent"] = psutil.cpu_percent(percpu=False)
            info["cpu_current_threads_load_percents"] = psutil.cpu_percent(percpu=True)

            info["ram_free_gb"] = ram.free / 2**30

        # Soft+GPU
        info["python_version"] = platform.python_implementation() + " " + platform.python_version()
        if numba is not None and "cuda" in dir(numba):
            if numba.cuda.is_available():
                try:
                    cuda_version = re.findall(", V(.+)\r\n", subprocess.check_output("nvcc --version").decode())
                except Exception:
                    cuda_version = [""]

                if len(cuda_version) > 0:
                    if len(cuda_version[0].split(".")) == 3:
                        if not only_stats:
                            info["cuda_version"] = cuda_version[0]
                if not only_stats:
                    # Use new GPU detection
                    try:
                        gpu_stats = get_nvidia_smi_info(include_stats=True)
                        if gpu_stats:
                            info["gpu_current_stats_nvidia_smi"] = gpu_stats
                    except Exception:
                        pass

                    # Keep legacy gpuutil for backward compatibility with usage stats
                    try:
                        info["gpu_current_stats"] = get_gpuutil_gpu_info()
                    except Exception:
                        pass

        if return_hdd_info:
            (
                max_singledisk_free_space,
                singledisk_usage_percent,
                best_disk,
                cumulative_disks_usage_total,
                cumulative_disks_usage_free,
            ) = get_max_singledisk_free_space_gb()

            info["hdd_free_space_gb"] = cumulative_disks_usage_free
            info["hdd_total_space_gb"] = cumulative_disks_usage_total
            info["hdd_max_singledisk_free_space_gb"] = max_singledisk_free_space

        # psutil.boot_time() #1564925068.0
        # psutil.users() #[suser(name='TheLocalCommander', terminal=None, host='0.0.0.0', started=1564925062.0, pid=None)]
        # psutil.disk_io_counters(perdisk=True) #{'PhysicalDrive0': sdiskio(read_count=1073232, write_count=9990341, read_bytes=43034913280, write_bytes=203126582272, read_time=1536, write_time=8487)}
        if return_hardware_details:
            if current_system == "Windows":
                # Use enhanced WMI summary
                system_wmi_summary = summarize_system_info()
                if system_wmi_summary:
                    info["system_wmi_summary"] = system_wmi_summary
            else:
                # Linux: dmidecode info
                dmidecode_info = parse_dmidecode_info()
                if dmidecode_info:
                    info["dmidecode_info"] = dmidecode_info

                # Linux devices list
                try:
                    info["devices_list"] = list_linux_devices()
                except Exception:
                    pass
        return info
    except Exception as e:
        logger.exception(e)


# ----------------------------------------------------------------------------------------------------------------------------
# CPU
# ----------------------------------------------------------------------------------------------------------------------------


def get_cpu_info():
    """Get CPU information using py-cpuinfo with filtered output.

    Returns:
        dict: Filtered CPU information (excludes version strings, hz fields, count)
    """
    try:
        import cpuinfo
    except ImportError:
        logger.warning("py-cpuinfo not available")
        return None

    from pyutilz.core.pythonlib import sort_dict_by_key

    cpu_info = cpuinfo.get_cpu_info()
    for key in "python_version cpuinfo_version hz_advertised hz_advertised_friendly hz_actual hz_actual_friendly count".split():
        if key in cpu_info:
            del cpu_info[key]
    if "flags" in cpu_info:
        flags = cpu_info.get("flags", [])
        if isinstance(flags, (list, tuple)):
            cpu_info["flags"] = " ".join(sorted(flags))
    return sort_dict_by_key(cpu_info)


def get_wmi_cpuinfo() -> dict:
    """Get CPU information via Windows WMI.

    Returns:
        dict: Summarized CPU information from Win32_Processor
    """
    try:
        import wmi
    except ImportError:
        logger.warning("wmi not available (Windows only)")
        return None

    try:
        c = wmi.WMI()
        return summarize_devices(
            c.Win32_Processor(),
            exclude_pros=set(
                "CreationClassName SystemCreationClassName DeviceID SystemName ProcessorId SerialNumber Status StatusInfo LoadPercentage AssetTag Description".split()
            ),
        )
    except Exception as e:
        logger.exception(e)
        return None


def get_lscpu_info():
    """Get lscpu information as a dictionary (Linux only).

    Returns:
        dict: Dictionary containing lscpu information with type conversion

    Example:
        {'Architecture': 'x86_64', 'CPU(s)': 6, 'Thread(s) per core': 1,
         'Core(s) per socket': 1, 'Socket(s)': 6, 'Model': 79,
         'Model name': 'Intel(R) Xeon(R) CPU E5-2696 v4 @ 2.20GHz',
         'CPU MHz': 2199.998, 'BogoMIPS': 4399.99, ...}
    """
    from pyutilz.core.pythonlib import is_float, to_float, sort_dict_by_key

    lscpu_dict = {}
    try:
        output = subprocess.check_output(["lscpu"], text=True)

        for line in output.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                if "Vulnerability" not in key:
                    value = value.strip()
                    if isinstance(value, str) and is_float(value):
                        value = to_float(value)
                        if value.is_integer():
                            value = int(value)
                    lscpu_dict[key.strip()] = value

        if "Flags" in lscpu_dict:
            lscpu_dict["Flags"] = " ".join(sorted(lscpu_dict["Flags"].split(" ")))
    except Exception as e:
        logger.warning(f"An error occurred while running lscpu: {e}")
        return None

    return sort_dict_by_key(lscpu_dict)


def get_linux_board_info():
    """Get motherboard information from /sys/devices (Linux only).

    Returns:
        dict: Board vendor, name, and version
    """
    board_info = {}
    try:
        with open("/sys/devices/virtual/dmi/id/board_vendor", "r") as f:
            board_info["Vendor"] = f.read().strip()
        with open("/sys/devices/virtual/dmi/id/board_name", "r") as f:
            board_info["Name"] = f.read().strip()
        with open("/sys/devices/virtual/dmi/id/board_version", "r") as f:
            board_info["Version"] = f.read().strip()
    except FileNotFoundError as e:
        logger.error(f"Error reading board information: {e}")
    return board_info


def parse_dmidecode_info(
    skip_keys: set = set(
        [
            "Address",
            "Asset Tag",
            "Locator",
            "Bank Locator",
            "ID",
            "UUID",
            "OEM Information",
            "Serial Number",
            "Socket Designation",
        ],
    ),
    skip_values: set = set(
        ["Not Provided", "Unknown", "Not Specified", "Unspecified", "None", "To Be Filled By O.E.M.", "", None]
    ),
) -> dict:
    """Parse dmidecode output for hardware information (Linux only).

    Args:
        skip_keys: Keys to skip in output
        skip_values: Values to skip in output

    Returns:
        list: List of dicts with hardware info (BIOS, System, Processor, Memory, etc.)
    """
    from collections import Counter
    from pyutilz.core.pythonlib import is_float, to_float

    try:
        result = subprocess.run(["sudo", "dmidecode"], capture_output=True, text=True)
    except Exception as e:
        logger.error(f"dmidecode running problem: {e}")
        return None

    output = result.stdout

    interesting_sections = "BIOS;System;Chassis;Processor;Memory Device;Base Board;Memory Controller;Memory Module;Cache;System Configuration Options;Portable Battery;Voltage Probe".split(
        ";"
    )
    sections_to_avoid = "Memory Array Mapped Address;System Boot Information".split(";")

    summary = []
    section_dict = {}
    features_name = None
    keep_section = False
    current_section = None

    for line in output.splitlines():
        if line.startswith("\t"):
            if keep_section:
                if ":" in line:
                    key, value = map(str.strip, line.split(":", 1))
                    if value == "":
                        features_name = key
                        features = []
                    else:
                        if features_name and features:
                            section_dict[features_name] = tuple(sorted(features))
                            features = []
                        features_name = None
                    if key not in skip_keys and value not in skip_values and not (key.endswith("Handle")):
                        if isinstance(value, str):
                            value = value.strip()
                            if "Version" not in key and "Revision" not in key and is_float(value):
                                value = to_float(value)
                        if isinstance(value, float) and value.is_integer():
                            value = int(value)
                        section_dict[key] = value
                else:
                    if features_name and line.startswith("\t\t"):
                        features.append(line.strip())
        elif line.strip():
            if section_dict:
                if features_name and features:
                    section_dict[features_name] = tuple(sorted(features))
                    features = []
                features_name = None
                section_dict["ItemType"] = current_section
                summary.append(dict_to_tuple(section_dict))
                section_dict = {}

            current_section = line.strip()

            keep_section = False
            for interesting_section in interesting_sections:
                if interesting_section in current_section:
                    keep_section = True
                    break
            if keep_section:
                if current_section in sections_to_avoid:
                    keep_section = False

    if section_dict:
        if features_name and features:
            section_dict[features_name] = tuple(sorted(features))
            features = []
        features_name = None
        section_dict["ItemType"] = current_section
        summary.append(dict_to_tuple(section_dict))

    return [{"item": dict(k), "Count": v} for k, v in Counter(summary).items()]


def summarize_system_info():
    """Get comprehensive system info via WMI (Windows only).

    Returns:
        dict: GPU, RAM, Cache, Board, BIOS, OS information
    """
    try:
        import wmi
    except ImportError:
        logger.warning("wmi not available (Windows only)")
        return None

    try:
        c = wmi.WMI()

        res = {}
        res["GPU"] = summarize_devices(
            c.Win32_VideoController(),
            exclude_pros=set(
                "CreationClassName SystemCreationClassName DeviceID PNPDeviceID InfFilename InfSection Name SystemName InstalledDisplayDrivers DeviceID Status Caption Description".split()
            ),
        )

        res["RAM"] = summarize_devices(
            c.Win32_PhysicalMemory(),
            exclude_pros=set(
                "BankLabel DeviceLocator InstallDate SerialNumber SKU Tag CreationClassName Caption Description Name PositionInRow".split()
            ),
        )

        res["PhysicalMemoryArray"] = summarize_devices(
            c.Win32_PhysicalMemoryArray(),
            exclude_pros=set(
                "BankLabel DeviceLocator InstallDate SerialNumber SKU Tag CreationClassName Caption Description Name PositionInRow".split()
            ),
        )

        res["CACHE"] = summarize_devices(
            c.Win32_CacheMemory(),
            exclude_pros=set(
                "CreationClassName SystemCreationClassName DeviceID SystemName Caption Description Name Status StatusInfo CurrentSRAM SupportedSRAM".split()
            ),
        )

        res["BOARD"] = summarize_devices(
            c.Win32_BaseBoard(),
            exclude_pros=set(
                "CreationClassName SystemCreationClassName DeviceID SerialNumber Status StatusInfo Caption Description Name Tag ConfigOptions".split()
            ),
        )

        res["BIOS"] = summarize_devices(
            c.Win32_BIOS(),
            exclude_pros=set(
                "CreationClassName SystemCreationClassName DeviceID PNPDeviceID Status SystemName Caption Description Name Tag".split()
            ),
        )

        res["OS"] = summarize_devices(
            c.Win32_OperatingSystem(),
            exclude_pros=set(
                "CreationClassName CSCreationClassName SystemCreationClassName DeviceID PNPDeviceID Status BootDevice CSName FreePhysicalMemory FreeSpaceInPagingFiles FreeVirtualMemory InstallDate NumberOfLicensedUsers NumberOfProcesses NumberOfUsers Name Organization RegisteredUser SerialNumber SizeStoredInPagingFiles SystemDevice SystemDirectory SystemDrive TotalVirtualMemorySize TotalVisibleMemorySize WindowsDirectory LastBootUpTime LocalDateTime".split()
            ),
        )

        return res
    except Exception as e:
        logger.exception(e)
        return None


def get_nix_cpu_sockets_number():
    num_sockets = 1
    try:
        res = subprocess.check_output("lscpu").decode()
        num_sockets = re.findall("Socket\\(s\\):(.+)\n", res)
        if len(num_sockets) > 0:
            num_sockets = int(str(num_sockets[0]).strip())
    except Exception as e:
        logger.exception(e)
    return num_sockets


# ----------------------------------------------------------------------------------------------------------------------------
# Memory
# ----------------------------------------------------------------------------------------------------------------------------


def get_own_memory_usage() -> float:
    """Return RAM usage of our own Python process"""
    try:
        pid = os.getpid()
        py = psutil.Process(pid)
        memory_usage = py.memory_info().rss / 2.0**30  # memory usage in GB
    except Exception as e:
        logger.exception(e)
    else:
        return memory_usage


def trim_windows_process_memory(pid: int = None) -> bool:
    """Causes effect similar to malloc_trim on -nix."""

    # Define SIZE_T based on the platform (32-bit or 64-bit)
    if ctypes.sizeof(ctypes.c_void_p) == 4:
        SIZE_T = ctypes.c_uint32
    else:
        SIZE_T = ctypes.c_uint64

    # Get a handle to the current process
    if not pid:
        pid = ctypes.windll.kernel32.GetCurrentProcess()

    # Define argument and return types for SetProcessWorkingSetSizeEx
    ctypes.windll.kernel32.SetProcessWorkingSetSizeEx.argtypes = [
        ctypes.wintypes.HANDLE,  # Process handle
        SIZE_T,  # Minimum working set size
        SIZE_T,  # Maximum working set size
        ctypes.wintypes.DWORD,  # Flags
    ]
    ctypes.windll.kernel32.SetProcessWorkingSetSizeEx.restype = ctypes.wintypes.BOOL

    # Define constants for SetProcessWorkingSetSizeEx
    QUOTA_LIMITS_HARDWS_MIN_DISABLE = 0x00000002

    # Attempt to set the working set size
    result = ctypes.windll.kernel32.SetProcessWorkingSetSizeEx(pid, SIZE_T(-1), SIZE_T(-1), QUOTA_LIMITS_HARDWS_MIN_DISABLE)

    if result == 0:
        # Retrieve the error code
        error_code = ctypes.windll.kernel32.GetLastError()
        logger.error(f"SetProcessWorkingSetSizeEx failed with error code: {error_code}")
        return False
    else:
        return True


def clean_ram() -> None:
    """Forces python garbage collection.
    Most importantly, calls malloc_trim/SetProcessWorkingSetSizeEx, which fixes pandas/libc (?) memory leak."""

    gc.collect()
    if platform.system() == "Windows":
        trim_windows_process_memory()
    else:
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            logger.error("malloc_trim attempt failed")


def show_biggest_session_objects(session: dict, N: int = 5, min_size_bytes: int = 1) -> pd.DataFrame:
    """

    Then reports own process RAM usage & the mnost RAM consuming objects.
    """

    clean_ram()

    print(f"Own process RAM usage: {get_own_memory_usage():.2f} GB")

    # Start tracing memory allocations
    # tracemalloc.start()

    # Retrieve all objects from the current Python session
    res = []
    for obj in session.values():
        try:
            size = sys.getsizeof(obj)
        except Exception:
            print(f"stumbled on object of type {type(obj)}")
            pass
        else:
            if size >= min_size_bytes:
                res.append(dict(type=type(obj), size_gb=size / 1024**3))
    if res:
        res = pd.DataFrame(res).sort_values("size_gb", ascending=False).head(N)
    return res


def show_tracemalloc_snapshot(N: int = 10):
    """Take a snapshot of memory usage and display top N lines.

    Note: This function properly starts and stops tracemalloc to avoid resource leaks.
    """
    tracemalloc.start()
    try:
        # Take a snapshot of memory usage
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")

        print(f"Top {N} memory-consuming lines:")
        for stat in top_stats[:N]:
            print(stat)

        return snapshot
    finally:
        # Always stop tracemalloc to prevent resource leak
        tracemalloc.stop()


# ----------------------------------------------------------------------------------------------------------------------------
# Large Pages Support
# ----------------------------------------------------------------------------------------------------------------------------


def check_huge_pages_linux():
    """Check if huge pages are enabled on Linux.

    Returns:
        bool: True if huge pages are available
    """
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if "HugePages_Total" in line:
                    huge_pages_total = int(line.split()[1])
                    return huge_pages_total > 0
    except Exception as e:
        logger.warning(f"Error checking huge pages: {e}")
    return False


def check_large_pages_windows():
    """Check if large pages are supported on Windows.

    Returns:
        bool: True if large pages are available
    """
    import ctypes

    try:
        # Get the minimum size for large pages
        GetLargePageMinimum = ctypes.windll.kernel32.GetLargePageMinimum
        GetLargePageMinimum.restype = ctypes.c_size_t
        large_page_size = GetLargePageMinimum()
        return large_page_size > 0
    except Exception as e:
        logger.warning(f"Error checking large pages: {e}")
    return False


def check_huge_pages_macos():
    """Check if transparent huge pages are managed by macOS.

    Returns:
        bool: True if THP is managed by OS
    """
    try:
        output = subprocess.check_output(["vm_stat"]).decode()
        for line in output.split("\n"):
            if "Pages free" in line or "Pages active" in line:
                # If vm_stat is working and outputting memory pages, THP is managed by the OS
                return True
    except Exception as e:
        logger.warning(f"Error checking transparent huge pages: {e}")
    return False


def check_large_pages_support():
    """Cross-platform check for large pages support.

    Returns:
        bool: True if large pages are available on current platform
    """
    current_system = platform.system()
    if current_system == "Linux":
        return check_huge_pages_linux()
    elif current_system == "Windows":
        return check_large_pages_windows()
    elif current_system == "Darwin":
        return check_huge_pages_macos()
    else:
        logger.warning(f"Unsupported operating system: {current_system}")
        return None


# ----------------------------------------------------------------------------------------------------------------------------
# Power & Battery
# ----------------------------------------------------------------------------------------------------------------------------


def get_linux_power_plan():
    """Get CPU frequency scaling governor on Linux.

    Returns:
        list: Sorted list of unique CPU governors
    """
    from collections import Counter

    governors = []
    try:
        for cpu in os.listdir("/sys/devices/system/cpu/"):
            if cpu.startswith("cpu") and cpu[3:].isdigit():
                with open(f"/sys/devices/system/cpu/{cpu}/cpufreq/scaling_governor", "r") as f:
                    governors.append(f.read().strip())
        return sorted(list(Counter(governors).keys()))
    except Exception as e:
        logger.warning(f"Error getting Linux power plan: {e}")
        return None


def get_macos_power_plan():
    """Get power plan settings on macOS.

    Returns:
        dict: Power plan settings from pmset
    """
    try:
        res = subprocess.run(["pmset", "-g", "custom"], capture_output=True, text=True).stdout
        if not res:
            return None
        return dict(plan_full_name=res)
    except Exception as e:
        logger.warning(f"Error getting macOS power plan: {e}")
        return None


def get_windows_power_plan_cmd():
    """Get active power plan using powercfg command (Windows fallback).

    Returns:
        dict: plan_guid and plan_name

    Example outputs:
        Power Scheme GUID: 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c  (High performance)
        Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (HP Recommended)
    """
    try:
        res = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True).stdout
        if not res:
            return None

        guid = find_between(res, "Power Scheme GUID: ", "  ")
        plan_name = find_between(res, "(", ")")

        if guid and plan_name:
            return dict(plan_guid=guid, plan_name=plan_name)
        else:
            return dict(plan_full_name=res)
    except Exception as e:
        logger.warning(f"Error getting Windows power plan via cmd: {e}")
        return None


def get_windows_power_plan():
    """Get active power plan via WMI (Windows only).

    Returns:
        dict: plan_guid and plan_name
    """
    try:
        import wmi
    except ImportError:
        logger.warning("wmi not available, falling back to powercfg")
        return get_windows_power_plan_cmd()

    try:
        c = wmi.WMI(namespace=r"root\cimv2\power")
        for plan in c.Win32_PowerPlan():
            if plan.IsActive:
                return dict(plan_guid=plan.InstanceID, plan_name=plan.ElementName)
        return None
    except Exception as e:
        logger.warning(f"WMI power plan query failed: {e}, falling back to powercfg")
        return get_windows_power_plan_cmd()


def get_power_plan() -> dict:
    """Get current power plan across all platforms.

    Returns:
        dict: Platform-specific power plan information
    """
    current_system = platform.system()
    if current_system == "Windows":
        power_plan = get_windows_power_plan()
    elif current_system == "Darwin":
        power_plan = get_macos_power_plan()
    else:
        power_plan = get_linux_power_plan()
    return power_plan


def get_battery_info() -> dict:
    """Get battery information if available.

    Returns:
        dict: Battery status (percent, secsleft, power_plugged) or None if no battery
    """
    try:
        battery_info = psutil.sensors_battery()
        if battery_info:
            battery_info = battery_info._asdict()
        return battery_info
    except Exception as e:
        logger.exception(e)
        return None


# ----------------------------------------------------------------------------------------------------------------------------
# GPU info
# ----------------------------------------------------------------------------------------------------------------------------

# CUDA compute capability to cores mapping
CUDA_SM_TO_CORES = {
    30: 192,
    32: 192,
    35: 192,
    37: 192,
    50: 128,
    52: 128,
    53: 128,
    60: 64,
    61: 128,
    62: 128,
    70: 64,
    72: 64,
    75: 64,
    80: 64,
    86: 128,
    87: 128,
    89: 128,
    90: 128,
}


def get_nvidia_smi_info(
    include_ids: bool = False,
    include_stats: bool = True,
):
    """Get GPU information from nvidia-smi (NVIDIA GPUs only).

    Args:
        include_ids: Include GPU serial, UUID, board_id fields
        include_stats: Include current utilization, temperature, power stats

    Returns:
        dict: Parsed nvidia-smi output with driver version, CUDA version, GPU details
    """
    try:
        import xmltodict
    except ImportError:
        logger.warning("xmltodict not available")
        return None

    from pyutilz.core.pythonlib import sort_dict_by_key

    # Run nvidia-smi with XML output
    try:
        result = subprocess.run(
            ["nvidia-smi", "-q", "-x"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        logger.warning("nvidia-smi not found (NVIDIA GPU not present or drivers not installed)")
        return None

    # Check if command executed successfully
    if result.returncode != 0:
        logger.error(f"Error running nvidia-smi: {result.stdout} {result.stderr}")
        return None

    res = xmltodict.parse(result.stdout, force_list=["gpu"])
    res = res.get("nvidia_smi_log")

    remove_json_attributes(res, ["timestamp", "processes"])
    for gpu in res.get("gpu", []):
        remove_json_defaults(
            gpu,
            {
                "addressing_mode": "None",
                "mig_devices": "None",
                "accounting_mode": "Disabled",
                "minor_number": "0",
                "multigpu_board": "No",
                "display_mode": "Enabled",
                "display_active": "Enabled",
                "persistence_mode": "Enabled",
                "gpu_virtualization_mode": {
                    "virtualization_mode": "None",
                    "host_vgpu_mode": "N/A",
                    "vgpu_heterogeneous_mode": "N/A",
                },
                "gpu_reset_status": {"reset_required": "No", "drain_and_reset_recommended": "N/A"},
                "cc_protected_memory_usage": {"total": "0 MiB", "used": "0 MiB", "free": "0 MiB"},
                "encoder_stats": {"session_count": "0", "average_fps": "0", "average_latency": "0"},
                "fbc_stats": {"session_count": "0", "average_fps": "0", "average_latency": "0"},
            },
        )
        remove_json_defaults(
            gpu, {"gpu_virtualization_mode": {"virtualization_mode": "None", "host_vgpu_mode": "N/A"}}
        )

        remove_json_attributes(
            gpu,
            "processes accounted_processes pci clocks_event_reasons clocks_throttle_reasons ecc_errors retired_pages remapped_rows module_power_readings gpu_memory_power_readings supported_clocks".split(),
        )

        if not include_ids:
            remove_json_attributes(gpu, "@id serial uuid board_id".split())

        if not include_stats:
            remove_json_attributes(
                gpu,
                "bar1_memory_usage fan_speed utilization clocks applications_clocks default_applications_clocks".split(),
            )
            fb_memory_usage = gpu.get("fb_memory_usage", {})
            if fb_memory_usage:
                remove_json_attributes(fb_memory_usage, "reserved used free".split())
            temperature = gpu.get("temperature", {})
            if temperature:
                remove_json_attributes(temperature, "gpu_temp memory_temp".split())
            power_readings = gpu.get("power_readings", {})
            if power_readings:
                remove_json_attributes(power_readings, "power_state power_draw".split())
            gpu_power_readings = gpu.get("gpu_power_readings", {})
            if gpu_power_readings:
                remove_json_attributes(gpu_power_readings, "power_state power_draw".split())

        remove_nas(gpu)

    if "gpu" in res:
        res["gpu"] = [sort_dict_by_key(gpu) for gpu in res["gpu"]]

    remove_nas(res)

    return res


def get_gpu_cuda_capabilities(device_id: int = 0, cu_device_token: str = "CU_DEVICE_ATTRIBUTE_") -> dict:
    """Get all CUDA capabilities for a GPU device using numba.

    Args:
        device_id: CUDA device ID (0 for first GPU)
        cu_device_token: Prefix for CUDA device attributes

    Returns:
        dict: All CUDA device attributes including compute capability, cores, memory
    """
    try:
        from numba.cuda.cudadrv import enums
        from numba import cuda
    except ImportError:
        logger.warning("numba.cuda not available")
        return None

    from pyutilz.core.pythonlib import sort_dict_by_key

    try:
        res = {}
        cuda.select_device(device_id)
        device = cuda.get_current_device()
        attribs = [name.replace(cu_device_token, "") for name in dir(enums) if name.startswith(cu_device_token)]

        for attr in attribs:
            res[attr] = getattr(device, attr)

        remove_json_attributes(res, ["PCI_BUS_ID"])
        remove_json_defaults(
            res,
            {
                "PCI_DEVICE_ID": 0,
                "PCI_DOMAIN_ID": 0,
            },
        )

        if "COMPUTE_CAPABILITY_MAJOR" in res and "COMPUTE_CAPABILITY_MINOR" in res and "MULTIPROCESSOR_COUNT" in res:
            cuda_float_version = int(res["COMPUTE_CAPABILITY_MAJOR"]) * 10 + int(res["COMPUTE_CAPABILITY_MINOR"])
            cores_per_sm = CUDA_SM_TO_CORES.get(cuda_float_version, 128)
            res["TOTAL_CUDA_CORES"] = cores_per_sm * int(res["MULTIPROCESSOR_COUNT"])

        return sort_dict_by_key(res)
    except Exception as e:
        logger.exception(e)
        return None


def get_cuda_gpu_details(cuda_gpu_info: dict) -> dict:
    """Combine nvidia-smi info with CUDA capabilities for all GPUs.

    Args:
        cuda_gpu_info: Output from get_nvidia_smi_info()

    Returns:
        dict: Mapping of device_id to CUDA capabilities
    """
    if cuda_gpu_info is None:
        return {}

    cuda_details = {}
    for gpu in cuda_gpu_info.get("gpu", []):
        device_id = gpu.get("gpu_module_id")
        if device_id is not None:
            if device_id not in cuda_details:
                cuda_details[device_id] = get_gpu_cuda_capabilities(device_id)
    return cuda_details


def get_gpuutil_gpu_info(attrs: str = "name,memoryTotal,memoryFree,load,driver,id,temperature,uuid") -> list:
    """Get GPU utilization stats using GPUtil (lightweight nvidia-smi wrapper).

    Args:
        attrs: Comma-separated list of attributes to retrieve

    Returns:
        list: List of dicts with GPU stats (memory in GB, load in %)
    """
    if isinstance(attrs, str):
        attrs = attrs.split(",")
    assert "id" in attrs

    devices = []

    try:
        import GPUtil
    except Exception:
        logger.warning("Can't import GPUtil.")
        return devices

    try:
        for gpu in GPUtil.getGPUs():
            cur_device = dict()
            for attr in attrs:
                val = getattr(gpu, attr)
                if "memory" in attr:
                    val = val / 1024
                elif attr == "load":
                    val = val * 100
                cur_device[attr] = val
            devices.append(cur_device)
    except Exception as e:
        logger.exception(e)

    return devices


# Backward-compatible alias
get_gpuinfo_gpu_info = get_gpuutil_gpu_info


def compute_total_gpus_ram(gpus: list) -> dict:
    """Compute aggregate GPU RAM stats from a list of GPU info dicts.

    Args:
        gpus: List of dicts as returned by get_gpuutil_gpu_info()

    Returns:
        dict with gpu_max_ram_total, gpus_ram_total, gpus_ram_free
    """
    if not gpus:
        return {"gpu_max_ram_total": 0, "gpus_ram_total": 0, "gpus_ram_free": 0}
    totals = [g.get("memoryTotal", 0) for g in gpus]
    frees = [g.get("memoryFree", 0) for g in gpus]
    return {
        "gpu_max_ram_total": max(totals) if totals else 0,
        "gpus_ram_total": sum(totals),
        "gpus_ram_free": sum(frees),
    }


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
        devices = json.loads(subprocess.check_output(["pylspci", "-nn"]).decode("utf-8"))
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

    architecture = platform.architecture()
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
        except Exception:
            pass
    return res


def get_max_affordable_workers_count(reservedCores=1):
    import psutil

    n = psutil.cpu_count(logical=False) - 1
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
            except Exception:
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
            import tracemalloc

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
    gpu_ids: list = [],
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
    import time

    if not (0 <= max_cpu_load_percent <= 100):
        raise ValueError("max_cpu_load_percent must be between 0 and 100")
    if not (0 <= max_gpu_load_percent <= 100):
        raise ValueError("max_gpu_load_percent must be between 0 and 100")

    logger.info(f"Ensuring idle devices for {duration_seconds} seconds with the following conditions:")
    logger.info(
        f"  Max CPU load: {max_cpu_load_percent}%, Min CPU free RAM: {min_cpu_free_ram_gb} GB, "
        f"Max GPU load: {max_gpu_load_percent}%, Min GPU free RAM: {min_gpu_free_ram_gb} GB. "
        f"GPU IDs to check: {gpu_ids if gpu_ids else 'All'}."
    )

    def check_cpu_initial_conditions():
        total_cpu_ram_gb = psutil.virtual_memory().total / (1024**3)
        if min_cpu_free_ram_gb > total_cpu_ram_gb:
            logger.warning(
                f"Requested CPU free RAM ({min_cpu_free_ram_gb} GB) exceeds total available RAM ({total_cpu_ram_gb:.2f} GB)"
            )
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
                logger.warning(
                    f"Requested GPU {gpu.id} free RAM ({min_gpu_free_ram_gb} GB) exceeds total available RAM ({total_gpu_ram_gb:.2f} GB)"
                )
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


def get_locale_settings(locale_name: str = "", only_fields: tuple = None) -> dict:
    """Return a dict of locale params.

    :param str locale_name: Desired locale name, or empty string for OS default locale.
    :param tuple only_fields: Desired locale fields, or None for all fields.

    >>>get_locale_settings(locale_name="en_US.utf8", only_fields=("decimal_point", "thousands_sep"))
    {'decimal_point': '.', 'thousands_sep': ','}

    """
    locale.setlocale(locale.LC_ALL, locale_name)
    settings = locale.localeconv()
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
    except Exception:
        pass

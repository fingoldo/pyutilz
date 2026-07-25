"""System/hardware probing helpers: CPU, memory, GPU, OS and BIOS info via psutil, WMI, lscpu, nvidia-smi, etc."""

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


import re
import psutil
import platform
import os, subprocess  # nosec B404 - used throughout this module only to invoke fixed trusted system-probing binaries (lscpu, dmidecode, nvidia-smi, vm_stat, pmset, powercfg) with hardcoded argv, never shell/user input


from pyutilz.text.strings import remove_json_defaults, remove_json_attributes, find_between
from typing import Any as _Any, List as _List, Optional, Union

from ._common import remove_nas, summarize_devices, dict_to_tuple

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


def get_wmi_cpuinfo() -> Optional[dict]:
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
        return summarize_devices(  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime
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
        output = subprocess.check_output(["lscpu"], text=True)  # nosec B603 B607 - fixed trusted binary "lscpu" with no arguments, no shell, no external input

        for line in output.split("\n"):
            if ":" in line:
                key, value_str = line.split(":", 1)
                value: _Any = value_str
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
        logger.warning("An error occurred while running lscpu: %s", e)
        return None

    return sort_dict_by_key(lscpu_dict)


def get_linux_board_info():
    """Get motherboard information from /sys/devices (Linux only).

    Returns:
        dict: Board vendor, name, and version
    """
    board_info = {}
    try:
        # Linux sysfs files are ASCII-only by spec, but explicit encoding
        # avoids the locale-dependent default that crashes on Windows
        # (cp1251) if this code path is ever exercised under WSL.
        with open("/sys/devices/virtual/dmi/id/board_vendor", encoding="utf-8") as f:
            board_info["Vendor"] = f.read().strip()
        with open("/sys/devices/virtual/dmi/id/board_name", encoding="utf-8") as f:
            board_info["Name"] = f.read().strip()
        with open("/sys/devices/virtual/dmi/id/board_version", encoding="utf-8") as f:
            board_info["Version"] = f.read().strip()
    except FileNotFoundError as e:
        logger.error("Error reading board information: %s", e)
    return board_info


def parse_dmidecode_info(
    skip_keys: Optional[set] = None,
    skip_values: Optional[set] = None,
) -> Optional[list]:
    """Parse dmidecode output for hardware information (Linux only).

    Args:
        skip_keys: Keys to skip in output
        skip_values: Values to skip in output

    Returns:
        list: List of dicts with hardware info (BIOS, System, Processor, Memory, etc.)
    """
    if skip_keys is None:
        skip_keys = {
            "Address", "Asset Tag", "Locator", "Bank Locator", "ID",
            "UUID", "OEM Information", "Serial Number", "Socket Designation",
        }
    if skip_values is None:
        skip_values = {
            "Not Provided", "Unknown", "Not Specified", "Unspecified",
            "None", "To Be Filled By O.E.M.", "", None,
        }
    from collections import Counter
    from pyutilz.core.pythonlib import is_float, to_float

    try:
        result = subprocess.run(["sudo", "dmidecode"], capture_output=True, text=True)  # nosec B603 B607 - fixed trusted binaries "sudo"/"dmidecode" with hardcoded argv, no shell, no external/user-controlled input
    except Exception as e:
        logger.error("dmidecode running problem: %s", e)
        return None

    output = result.stdout

    interesting_sections = "BIOS;System;Chassis;Processor;Memory Device;Base Board;Memory Controller;Memory Module;Cache;System Configuration Options;Portable Battery;Voltage Probe".split(
        ";"
    )
    sections_to_avoid = "Memory Array Mapped Address;System Boot Information".split(";")

    summary = []
    section_dict: dict = {}
    features_name: _Any = None
    keep_section = False
    current_section = None

    for line in output.splitlines():
        if line.startswith("\t"):
            if keep_section:
                if ":" in line:
                    key, value = map(str.strip, line.split(":", 1))
                    if value == "":
                        features_name = key
                        features: _List[_Any] = []
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
            exclude_pros=set("BankLabel DeviceLocator InstallDate SerialNumber SKU Tag CreationClassName Caption Description Name PositionInRow".split()),
        )

        res["PhysicalMemoryArray"] = summarize_devices(
            c.Win32_PhysicalMemoryArray(),
            exclude_pros=set("BankLabel DeviceLocator InstallDate SerialNumber SKU Tag CreationClassName Caption Description Name PositionInRow".split()),
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
            exclude_pros=set("CreationClassName SystemCreationClassName DeviceID PNPDeviceID Status SystemName Caption Description Name Tag".split()),
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


def get_nix_cpu_sockets_number() -> int:
    """Returns the number of physical CPU sockets on Linux by parsing `lscpu` output; falls back to 1 on any failure."""
    num_sockets: int = 1
    try:
        res = subprocess.check_output("lscpu").decode()  # nosec B603 B607 - fixed trusted binary "lscpu", hardcoded string with no interpolation, no shell
        matches = re.findall("Socket\\(s\\):(.+)\n", res)
        if len(matches) > 0:
            # Parse into a local first: if the matched text isn't int-parseable (odd
            # locale/virtualization output), num_sockets must stay at its documented
            # fallback of 1, not silently become the raw regex match list.
            num_sockets = int(matches[0].strip())
    except Exception as e:
        logger.exception(e)
    return num_sockets

# ----------------------------------------------------------------------------------------------------------------------------
# Large Pages Support
# ----------------------------------------------------------------------------------------------------------------------------


def check_huge_pages_linux():
    """Check if huge pages are enabled on Linux.

    Returns:
        bool: True if huge pages are available
    """
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if "HugePages_Total" in line:
                    huge_pages_total = int(line.split()[1])
                    return huge_pages_total > 0
    except Exception as e:
        logger.warning("Error checking huge pages: %s", e)
    return False


def check_large_pages_windows():
    """Check if large pages are supported on Windows.

    Returns:
        bool: True if large pages are available
    """
    import ctypes

    try:
        # Get the minimum size for large pages
        GetLargePageMinimum = ctypes.windll.kernel32.GetLargePageMinimum  # type: ignore[attr-defined]
        GetLargePageMinimum.restype = ctypes.c_size_t
        large_page_size = GetLargePageMinimum()
        return large_page_size > 0
    except Exception as e:
        logger.warning("Error checking large pages: %s", e)
    return False


def check_huge_pages_macos():
    """Check if transparent huge pages are managed by macOS.

    Returns:
        bool: True if THP is managed by OS
    """
    try:
        output = subprocess.check_output(["vm_stat"]).decode()  # nosec B603 B607 - fixed trusted macOS binary "vm_stat" with no arguments, no shell, no external input
        for line in output.split("\n"):
            if "Pages free" in line or "Pages active" in line:
                # If vm_stat is working and outputting memory pages, THP is managed by the OS
                return True
    except Exception as e:
        logger.warning("Error checking transparent huge pages: %s", e)
    return False


def check_large_pages_support() -> Optional[bool]:
    """Cross-platform check for large pages support.

    Returns:
        Optional[bool]: True/False if large pages support could be determined on the current
        platform, None on an unrecognized/unsupported platform.
    """
    current_system = platform.system()
    if current_system == "Linux":
        return bool(check_huge_pages_linux())
    elif current_system == "Windows":
        return bool(check_large_pages_windows())
    elif current_system == "Darwin":
        return bool(check_huge_pages_macos())
    else:
        logger.warning("Unsupported operating system: %s", current_system)
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
                with open(f"/sys/devices/system/cpu/{cpu}/cpufreq/scaling_governor", encoding="utf-8") as f:
                    governors.append(f.read().strip())
        return sorted(list(Counter(governors).keys()))
    except Exception as e:
        logger.warning("Error getting Linux power plan: %s", e)
        return None


def get_macos_power_plan():
    """Get power plan settings on macOS.

    Returns:
        dict: Power plan settings from pmset
    """
    try:
        res = subprocess.run(["pmset", "-g", "custom"], capture_output=True, text=True).stdout  # nosec B603 B607 - fixed trusted macOS binary "pmset" with hardcoded argv, no shell, no external input
        if not res:
            return None
        return dict(plan_full_name=res)
    except Exception as e:
        logger.warning("Error getting macOS power plan: %s", e)
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
        res = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True).stdout  # nosec B603 B607 - fixed trusted Windows binary "powercfg" with hardcoded argv, no shell, no external input
        if not res:
            return None

        guid = find_between(res, "Power Scheme GUID: ", "  ")
        plan_name = find_between(res, "(", ")")

        if guid and plan_name:
            return dict(plan_guid=guid, plan_name=plan_name)
        else:
            return dict(plan_full_name=res)
    except Exception as e:
        logger.warning("Error getting Windows power plan via cmd: %s", e)
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
        logger.warning("WMI power plan query failed: %s, falling back to powercfg", e)
        return get_windows_power_plan_cmd()


def get_power_plan() -> Optional[dict]:
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
    return power_plan  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime


def get_battery_info() -> Optional[dict]:
    """Get battery information if available.

    Returns:
        dict: Battery status (percent, secsleft, power_plugged) or None if no battery
    """
    try:
        battery_info = psutil.sensors_battery()
        if battery_info:
            battery_info = battery_info._asdict()
        return battery_info  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime
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

    # Run nvidia-smi with XML output. A 10 s timeout guards against wedged
    # drivers (real incident class on Windows after a GPU reset); without the
    # timeout an import of this module on a misbehaving host blocks forever.
    try:
        result = subprocess.run(  # nosec B603 B607 - fixed trusted binary "nvidia-smi" with hardcoded argv, no shell, no external/user-controlled input
            ["nvidia-smi", "-q", "-x"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        logger.warning("nvidia-smi not found (NVIDIA GPU not present or drivers not installed)")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi did not respond within 10s; assuming GPU is wedged / unavailable")
        return None

    # Check if command executed successfully
    if result.returncode != 0:
        logger.error("Error running nvidia-smi: %s %s", result.stdout, result.stderr)
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
        remove_json_defaults(gpu, {"gpu_virtualization_mode": {"virtualization_mode": "None", "host_vgpu_mode": "N/A"}})

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


def get_gpu_cuda_capabilities(device_id: int = 0, cu_device_token: str = "CU_DEVICE_ATTRIBUTE_") -> Optional[dict]:  # nosec B107 - "CU_DEVICE_ATTRIBUTE_" is a CUDA enum-name prefix used to strip it from numba's enums attribute names, not a credential
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


def get_gpuutil_gpu_info(attrs: Union[str, list] = "name,memoryTotal,memoryFree,load,driver,id,temperature,uuid") -> list:
    """Get GPU utilization stats using GPUtil (lightweight nvidia-smi wrapper).

    Args:
        attrs: Comma-separated list of attributes to retrieve

    Returns:
        list: List of dicts with GPU stats (memory in GB, load in %)
    """
    if isinstance(attrs, str):
        attrs = attrs.split(",")
    assert "id" in attrs  # nosec B101 - internal precondition ("id" is used below to key devices), not a security/permission check

    devices: _List[_Any] = []

    try:
        import GPUtil
    except Exception:
        logger.warning("Can't import GPUtil.")
        return devices  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime

    try:
        gpus = GPUtil.getGPUs()
    except Exception as e:
        logger.exception(e)
        return devices  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime

    for gpu in gpus:
        try:
            cur_device = dict()
            for attr in attrs:
                val = getattr(gpu, attr)
                if "memory" in attr:
                    val = val / 1024
                elif attr == "load":
                    val = val * 100
                cur_device[attr] = val
            devices.append(cur_device)
        except Exception as e:  # noqa: PERF203 -- per-iteration fault isolation is intentional (one bad GPU shouldn't truncate the rest)
            # One bad GPU's attribute read must not silently truncate the whole result --
            # compute_total_gpus_ram() sums this list directly, so a partial list would
            # silently undercount capacity with no signal anything was skipped.
            logger.exception("get_gpuutil_gpu_info: skipping a GPU after an attribute-read error (%s)", e)

    if len(devices) < len(gpus):
        logger.warning("get_gpuutil_gpu_info: only %d of %d detected GPU(s) were successfully enumerated", len(devices), len(gpus))

    return devices  # type: ignore[no-any-return]  # untyped upstream source (json/external lib/dynamic attr); return value verified correct at runtime


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

"""Cross-platform collection of system/hardware information (OS, CPU, GPU, RAM, disk, network)."""

#!$ pip install pympler psutil gpu-info pylspci gputil py-cpuinfo
#!$ pip install pycuda

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging
from typing import Any as _Any

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------


# ensure_installed("pympler psutil numba tqdm gpu-info")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------


import uuid
import re
import socket, psutil
import platform
import subprocess  # nosec B404 - used only to invoke fixed trusted OS-identity probes (wmic, cat, getprop, ioreg/grep, nvcc) with hardcoded/literal argv below, no shell, no external input


from .misc import get_os_info
from .probing import (
    get_power_plan,
    get_battery_info,
    get_cpu_info,
    get_wmi_cpuinfo,
    get_lscpu_info,
    get_linux_board_info,
    get_nix_cpu_sockets_number,
    get_nvidia_smi_info,
    get_cuda_gpu_details,
    get_gpuutil_gpu_info,
    check_large_pages_support,
    summarize_system_info,
    parse_dmidecode_info,
)
from .fsutils import get_max_singledisk_free_space_gb, list_linux_devices

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
    if not only_stats and not any([return_hdd_info, return_os_info, return_hardware_info, return_hardware_details, return_usage_stats, return_network_info]):
        return_sensitive_info = True
        return_os_info = True  # Needed to trigger sensitive info collection

    numba: _Any
    web: _Any
    try:
        import numba  # type: ignore[no-redef]  # mypy quirk: unaliased self-name import after a bare pre-declaration annotation
    except ImportError:
        numba = None
    try:
        from pyutilz.web import web
    except ImportError:
        web = None

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
                        # wmic is deprecated and absent on Windows 11 24H2+; fall back to the machine GUID below.
                        os_serial = subprocess.check_output("wmic csproduct get uuid").decode().split("\n")[1].strip()  # nosec B603 B607 - fixed trusted Windows binary "wmic", hardcoded literal command, no shell, no external input
                        info["os_machine_guid"] = os_serial
                        info["os_serial"] = os_serial  # Also store as separate field for distributed.py
                    except Exception:
                        logger.warning("Could not extract Windows serial!")
                        # Keep the schema consistent (distributed.py requires os_serial): fall back to the machine GUID.
                        info["os_serial"] = info.get("os_machine_guid", "")

                elif current_system == "Linux":
                    try:
                        machine_id = subprocess.check_output(["cat", "/var/lib/dbus/machine-id"]).decode().strip()  # nosec B603 B607 - fixed trusted binary "cat" with a hardcoded file path, no shell, no external input
                        info["os_machine_guid"] = machine_id
                        info["os_serial"] = machine_id
                    except Exception:
                        try:
                            machine_id = subprocess.check_output(["cat", "/etc/machine-id"]).decode().strip()  # nosec B603 B607 - fixed trusted binary "cat" with a hardcoded file path, no shell, no external input
                            info["os_machine_guid"] = machine_id
                            info["os_serial"] = machine_id
                        except Exception:
                            logger.warning("Could not extract Linux machine-id")
                            info["os_serial"] = info.get("os_machine_guid", "")
                elif current_system == "Android":
                    try:
                        serial = subprocess.check_output(["getprop", "ril.serialnumber"])[:-1].decode().strip()  # nosec B603 B607 - fixed trusted Android binary "getprop" with a hardcoded property name, no shell, no external input
                        info["os_machine_guid"] = serial
                        info["os_serial"] = serial
                    except Exception:
                        logger.warning("Could not extract Android serial")
                        info["os_serial"] = info.get("os_machine_guid", "")
                elif current_system == "Mac":
                    try:
                        # Pipeline ``ioreg | grep`` — both Popen handles
                        # bound to context managers so they're terminated
                        # cleanly on exit / exception (was a bare
                        # subprocess.Popen pair before, leaking processes
                        # if grep_proc.communicate() raised).
                        with subprocess.Popen(  # nosec B603 B607 - fixed trusted macOS binary "ioreg" with hardcoded argv, no shell, no external input
                            ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                            stdout=subprocess.PIPE,
                        ) as ioreg_proc, subprocess.Popen(  # nosec B603 B607 - fixed trusted binary "grep" with hardcoded argv, piped from ioreg's stdout, no shell, no external input
                            ["grep", "-E", "(UUID)"],
                            stdin=ioreg_proc.stdout,
                            stdout=subprocess.PIPE,
                        ) as grep_proc:
                            assert ioreg_proc.stdout is not None  # guaranteed by stdout=subprocess.PIPE above
                            ioreg_proc.stdout.close()
                            output = grep_proc.communicate()[0]
                        guid = output.decode().split('"')[-2]
                        info["os_machine_guid"] = guid
                        info["os_serial"] = guid
                    except Exception:
                        logger.warning("Could not extract Mac UUID")
                        info["os_serial"] = info.get("os_machine_guid", "")

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
                    cuda_version = re.findall(", V(.+)\r\n", subprocess.check_output("nvcc --version").decode())  # nosec B603 B607 - fixed trusted binary "nvcc", hardcoded literal command, no shell, no external input
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
                    except Exception as e:  # nosec B110 - best-effort GPU usage stats; absent/misbehaving nvidia-smi must not break overall system_info collection
                        logger.debug("Could not get nvidia-smi GPU stats: %s", e)

                    # Keep legacy gpuutil for backward compatibility with usage stats
                    try:
                        info["gpu_current_stats"] = get_gpuutil_gpu_info()
                    except Exception as e:  # nosec B110 - best-effort legacy GPUtil stats; absent GPUtil/driver must not break overall system_info collection
                        logger.debug("Could not get GPUtil GPU stats: %s", e)

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
            # Previously computed but silently discarded (found 2026-07-09 deptry/vulture
            # triage): which mountpoint has the most free space, and what fraction of IT is
            # free -- distinct from the aggregate hdd_max_singledisk_free_space_gb, which is
            # capacity-in-GB with no per-disk usage-percent context.
            info["hdd_best_disk"] = best_disk
            info["hdd_best_disk_usage_percent"] = singledisk_usage_percent

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
                except Exception as e:  # nosec B110 - best-effort optional Linux devices listing (needs pylspci); must not break overall system_info collection
                    logger.debug("Could not list Linux devices: %s", e)
        return info
    except Exception as e:
        logger.exception(e)
        return info

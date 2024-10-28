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

from .pythonlib import ensure_installed

# ensure_installed("pympler psutil numba tqdm gpu-info")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

import uuid
import locale
import pandas as pd
import re, json, tqdm
import socket, psutil
from pympler import asizeof
import platform, sys, importlib
import os, platform, subprocess
from datetime import timezone, datetime, timedelta

import gc
import sys
import ctypes
import tracemalloc

from .strings import remove_json_defaults

# ----------------------------------------------------------------------------------------------------------------------------
# IPython
# ----------------------------------------------------------------------------------------------------------------------------


def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


# ----------------------------------------------------------------------------------------------------------------------------
# Progressbars
# ----------------------------------------------------------------------------------------------------------------------------


def tqdmu(*args, **kwargs):

    if run_from_ipython():
        try:
            res = tqdm.tqdm_notebook(*args, **kwargs)
        except Exception as e:
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
    """
    return_sensitive_info=True also returns machine GUID and host name.
    """
    # ensure_installed("x86cpu py-cpuinfo")

    import numba
    from numba import cuda
    from . import web, strings

    try:
        import x86cpu
    except Exception as e:
        pass
    try:
        import cpuinfo
    except Exception as e:
        pass

    info = dict()
    try:
        current_system = platform.system()
        if return_os_info:
            # Os
            info["os_system"] = current_system
            info["os_version"] = platform.version()
            info["os_platform"] = platform.platform()
            if current_system == "Windows":
                info["os_edition"] = platform.win32_edition()

            if return_sensitive_info:

                # (Almost) Constant features
                info["os_machine_guid"] = str(uuid.UUID(int=uuid.getnode()))
                if current_system == "Windows":
                    os_serial = None
                    try:
                        os_serial = subprocess.check_output("wmic csproduct get uuid").decode().split("\n")[1].strip()
                        info["os_machine_guid"] = os_serial
                    except:
                        logger.warning(f"Could not extract Windows serial!")

                elif current_system == "Linux":
                    info["os_machine_guid"] = subprocess.check_output("cat /var/lib/dbus/machine-id")
                elif current_system == "Android":
                    info["os_machine_guid"] = subprocess.check_output(["getprop", "ril.serialnumber"])[:-1]
                elif current_system == "Mac":
                    info["os_machine_guid"] = subprocess.check_output("ioreg -rd1 -c IOPlatformExpertDevice | grep -E '(UUID)'", shell=True).split('"')[-2]

                if return_network_info:
                    info["host_name"] = socket.gethostname()
                    info["host_external_ip"] = web.get_external_ip()

        if return_hardware_info or return_usage_stats:
            cpu_freq = psutil.cpu_freq()
            ram = psutil.virtual_memory()

        if return_hardware_info:
            # CPU

            cpu_info = cpuinfo.get_cpu_info()

            for key in "python_version cpuinfo_version cpuinfo_version_string hz_advertised_friendly hz_actual_friendly".split():
                if key in cpu_info:
                    del cpu_info[key]
            info["cpu"] = cpu_info

            if current_system != "Windows":
                info["cpu_num_sockets"] = get_nix_cpu_sockets_number()
            info["cpu_num_cores"] = psutil.cpu_count(logical=False)
            info["cpu_num_threads"] = psutil.cpu_count(logical=True)

            info["cpu_min_frequency_hz"] = cpu_freq.min
            info["cpu_max_frequency_hz"] = cpu_freq.max

            # RAM
            info["ram_total_gb"] = ram.total / 2**30

        # info['os_user_name']=os.environ.get('USER')
        # info['os_user_name']=os.environ.get('USERNAME')

        if return_usage_stats:

            info["cpu_current_frequency_hz"] = cpu_freq.current
            info["cpu_current_load_percent"] = psutil.cpu_percent(percpu=False)
            info["cpu_current_threads_load_percents"] = psutil.cpu_percent(percpu=True)

            info["ram_free_gb"] = ram.free / 2**30

        # Soft+GPU
        info["python_version"] = platform.python_implementation() + " " + platform.python_version()
        if "cuda" in dir(numba):
            if numba.cuda.is_available():
                try:
                    cuda_version = re.findall(", V(.+)\r\n", subprocess.check_output("nvcc --version").decode())
                except:
                    cuda_version = [""]

                if len(cuda_version) > 0:
                    if len(cuda_version[0].split(".")) == 3:
                        if not only_stats:
                            info["cuda_version"] = cuda_version[0]
                if not only_stats:
                    try:
                        info["gpus_ram_total_gb"], info["gpus_ram_free_gb"], info["gpu_features"] = get_gpuinfo_gpu_info()
                    except:
                        pass
                    try:
                        info["gpu_num_devices"] = len(info["gpu_features"])
                    except:
                        pass
                    """
                    try:
                        info["gpu_additional_features"] = get_pycuda_gpu_info()
                    except:
                        pass
                    """
                try:
                    info["gpu_current_stats"] = get_gpuutil_gpu_info()
                except:
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
                wmi = None
                try:
                    # ensure_installed("pypiwin32")

                    import wmi
                except:
                    pass
                if wmi is not None:
                    # WMI
                    # full list: 'win32_bios,win32_systembios,win32_baseboard,win32_systemslot,win32_systemenclosure,win32_operatingsystem,win32_computersystem,win32_computersystemproduct,win32_videocontroller,win32_videoconfiguration,win32_diskdrive,win32_diskpartition,win32_logicaldisk,win32_networkadapter,win32_onboarddevice,win32_physicalmemory,win32_processor,win32_timezone'.split(','):
                    q = wmi.WMI()
                    for (
                        group
                    ) in "win32_bios,win32_baseboard,win32_operatingsystem,win32_computersystem,win32_computersystemproduct,win32_videocontroller,win32_physicalmemory,win32_processor,win32_timezone".split(
                        ","
                    ):

                        objs = []
                        for obj in getattr(q, group)():
                            obj_dict = dict()
                            for prop in obj.properties:
                                obj_dict[prop] = getattr(obj, prop)
                            objs.append(obj_dict)

                            if group == "win32_computersystem":
                                if "NumberOfProcessors" in obj.properties:
                                    if isinstance(obj.NumberOfProcessors, int):
                                        info["cpu_num_processors"] = obj.NumberOfProcessors
                        info[group] = objs
                else:
                    try:
                        info["devices_list"] = list_linux_devices()
                    except:
                        pass
        return info
    except Exception as e:
        logger.exception(e)


# ----------------------------------------------------------------------------------------------------------------------------
# CPU
# ----------------------------------------------------------------------------------------------------------------------------


def get_nix_cpu_sockets_number():
    num_sockets = 1
    try:
        res = subprocess.check_output("lscpu").decode()
        num_sockets = re.findall("Socket\(s\):(.+)\n", res)
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


def trim_memory():
    try:
        ctypes.windll.kernel32.SetProcessWorkingSetSize(
            ctypes.windll.kernel32.GetCurrentProcess(), -1, -1  # Minimum size; use -1 to let OS decide  # Maximum size; use -1 to let OS decide
        )
    except Exception as e:
        logger.warning("SetProcessWorkingSetSize attempt failed")


def clean_ram() -> None:
    """
    Forces python garbage collection.
    Most importantly, calls malloc_trim, which fixes pandas memory leak."""

    gc.collect()
    if platform.system() == "Windows":
        trim_memory()
    else:
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception as e:
            logger.warning("malloc_trim attempt failed")


def show_biggest_session_objects(N: int = 5, min_size_bytes: int = 1) -> pd.DataFrame:
    """

    Then reports own process RAM usage & the mnost RAM consuming objects.
    """

    clean_ram()

    p = psutil.Process()
    print(f"Own process RAM usage: {get_own_memory_usage():.2f} GB")

    # Start tracing memory allocations
    # tracemalloc.start()

    # Retrieve all objects from the current Python session
    res = []
    for obj in globals().values():
        try:
            size = sys.getsizeof(obj)
            if size >= min_size_bytes:
                res.append(dict(type=type(obj), size_gb=size / 1024**3))
        except Exception as e:
            # print(f"stumbled on object of type {type(obj)}")
            pass

    res = pd.DataFrame(res).sort_values("size_gb", ascending=False).head(N)
    return res


def show_tracemalloc_snapshot(N: int = 10):

    tracemalloc.start()

    # Take a snapshot of memory usage
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")

    print(f"Top {N} memory-consuming lines:")
    for stat in top_stats[:N]:
        print(stat)


# ----------------------------------------------------------------------------------------------------------------------------
# GPU info
# ----------------------------------------------------------------------------------------------------------------------------


def compute_total_gpus_ram(gpus: Sequence) -> dict:
    """Returns GPUs VRAM, total all devices & max, in GBs
    >>>compute_total_gpus_ram(get_gpuinfo_gpu_info())
    {'gpus_ram_total': 7.841796875,
    'gpus_ram_free': 7.1611328125,
    'gpu_max_ram_free': 7.1611328125,
    'gpu_max_ram_total': 7.841796875}
    """

    gpus_ram_total = 0.0
    gpus_ram_free = 0.0
    gpu_max_ram_free = 0.0
    gpu_max_ram_total = 0.0

    for gpu in gpus:
        free_mem = gpu.get("free_memory", 0) / 1024
        total_mem = gpu.get("used_memory", 0) / 1024 + free_mem
        gpus_ram_total += total_mem
        gpus_ram_free += free_mem
        if total_mem > gpu_max_ram_total:
            gpu_max_ram_total = total_mem
        if free_mem > gpu_max_ram_free:
            gpu_max_ram_free = free_mem

    return dict(gpus_ram_total=gpus_ram_total, gpus_ram_free=gpus_ram_free, gpu_max_ram_free=gpu_max_ram_free, gpu_max_ram_total=gpu_max_ram_total)


def get_gpuinfo_gpu_info() -> list:

    devices = []

    try:
        from gpuinfo.nvidia import get_gpus
    except Exception as e:
        logger.warning("Can't import get_gpus from gpuinfo.nvidia.")
        return devices

    try:
        gpus = get_gpus()
    except Exception as e:
        logger.warning("Nvidia GPUs not detected")
    else:
        for gpu in gpus:
            cur_device = gpu.__dict__
            cur_device.update(gpu.get_max_clock_speeds())
            cur_device.update(gpu.get_clock_speeds())
            mem = gpu.get_memory_details()
            cur_device.update(mem)
            devices.append(cur_device)

    return devices


def get_gpuutil_gpu_info(attrs: str = "name,memoryTotal,memoryFree,load,driver,id,temperature,uuid") -> list:
    # ensure_installed("gputil")
    if isinstance(attrs, str):
        attrs = attrs.split(",")
    assert "id" in attrs

    devices = []

    try:
        import GPUtil as GPU
    except Exception as e:
        logger.warning("Can't import GPUtil.")
        return devices

    try:
        for gpu in GPU.getGPUs():
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


def get_pycuda_gpu_info() -> list:
    # ensure_installed("pycuda")

    devices = []

    try:
        import pycuda.autoinit
        import pycuda.driver as cuda
    except Exception as e:
        logger.warning("Can't import pycuda.")
        return devices

    # free,total=cuda.mem_get_info()
    # print(free,total,"Global memory occupancy: %f%% free"%(free*100/total))

    try:
        for devicenum in range(cuda.Device.count()):
            device = cuda.Device(devicenum)
            cur_device = dict()
            cur_device["compute_capability"] = ".".join(map(str, device.compute_capability()))
            cur_device["total_memory"] = device.total_memory()
            attrs = device.get_attributes()
            features = dict()
            for key, value in attrs.items():
                features[str(key)] = value
            cur_device["features"] = features
            devices.append(cur_device)
    except Exception as e:
        logger.exception(e)

    return devices


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


def get_libs_versions(libs: Union[Sequence, str] = "numpy pandas numba", sep: str = " ") -> dict:

    if isinstance(libs, str):
        libs = libs.split(sep)

    res = {}

    for modulename in libs:
        try:
            tmp = importlib.import_module(modulename)
            version = sys.modules[modulename].__version__
            res[modulename] = version
        except Exception as e:
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
            except:
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
                report = report + "\n" + "{}: {:,} MB".format(name, size_mb).replace(",", "_")
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
# Locales, Date/Time, etc.
# ----------------------------------------------------------------------------------------------------------------------------


def get_utc_unix_timestamp():
    from datetime import timezone, datetime, timedelta

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
    except Exception as e:
        pass

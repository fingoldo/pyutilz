#!$ pip install pympler psutil x86cpu gpuinfo pylspci gputil py-cpuinfo
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

ensure_installed("pympler psutil numba tqdm")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

import uuid
import locale
import re, json, tqdm
import socket, psutil
import os, platform, subprocess
from datetime import timezone, datetime, timedelta


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
        return tqdm.tqdm_notebook(*args, **kwargs)
    else:
        return tqdm.tqdm(*args, **kwargs)


def get_system_info(
    return_os_info: bool = False,
    return_software_info: bool = False,
    return_hardware_info: bool = False,
    return_hardware_details: bool = False,
    return_usage_stats: bool = False,
    return_network_info: bool = False,
) -> dict:

    ensure_installed("x86cpu py-cpuinfo")

    import numba
    from numba import cuda
    from . import web, string

    import x86cpu, cpuinfo

    info = dict()
    try:
        current_system = platform.system()
        if return_os_info:
            # (Almost) Constant features
            info["os_machine_guid"] = str(uuid.UUID(int=uuid.getnode()))
            # Os
            info["os_system"] = current_system
            os_serial = None
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

        if return_software_info:
            info["os_platform"] = platform.platform()
            info["os_version"] = platform.version()

        if return_network_info:
            info["host_name"] = socket.gethostname()
            info["ip"] = web.get_external_ip()

        if return_hardware_info or return_usage_stats:
            cpu_freq = psutil.cpu_freq()
            ram = psutil.virtual_memory()

        if return_hardware_info:
            # CPU
            info["cpu_features"] = string.jsonize_atrtributes(x86cpu.info, exclude="reg0,reg1,reg7,report,report_template".split(","))
            info["cpu_additional_features"] = cpuinfo.get_cpu_info()

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
                    try:
                        info["gpu_additional_features"] = get_pycuda_gpu_info()
                    except:
                        pass
                try:
                    info["gpu_current_stats"] = get_gpu_util_stats()
                except:
                    pass

        (
            max_singledisk_free_space,
            singledisk_usage_percent,
            best_disk,
            cumulative_disks_usage_total,
            cumulative_disks_usage_free,
        ) = get_max_singledisk_free_space_gb()

        info["hdd_total_space_gb"] = cumulative_disks_usage_total
        info["hdd_free_space_gb"] = cumulative_disks_usage_free
        info["hdd_max_singledisk_free_space_gb"] = max_singledisk_free_space

        # psutil.boot_time() #1564925068.0
        # psutil.users() #[suser(name='TheLocalCommander', terminal=None, host='0.0.0.0', started=1564925062.0, pid=None)]
        # psutil.disk_io_counters(perdisk=True) #{'PhysicalDrive0': sdiskio(read_count=1073232, write_count=9990341, read_bytes=43034913280, write_bytes=203126582272, read_time=1536, write_time=8487)}
        if return_hardware_details:
            if current_system == "Windows":
                wmi = None
                try:
                    ensure_installed("pypiwin32")

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
        memory_usage = py.memory_info()[0] / 2.0**30  # memory usage in GB
    except Exception as e:
        logger.exception(e)
    else:
        return memory_usage


# ----------------------------------------------------------------------------------------------------------------------------
# GPU info
# ----------------------------------------------------------------------------------------------------------------------------


def get_gpuinfo_gpu_info():
    ensure_installed("gpuinfo")

    from gpuinfo.nvidia import get_gpus

    devices = dict()
    gpus_ram_total_gb = 0
    gpus_ram_free_gb = 0
    try:
        gpus = get_gpus()
    except Exception as e:
        logger.warning("Nvidia GPUs not detected")
    else:
        for gpu in gpus:
            cur_device = gpu.__dict__
            gpu_name = cur_device["name"]
            del cur_device["name"]
            cur_device.update(gpu.get_max_clock_speeds())
            cur_device.update(gpu.get_clock_speeds())
            mem = gpu.get_memory_details()
            gpus_ram_total_gb = gpus_ram_total_gb + mem["used_memory"] / 1024 + mem["free_memory"] / 1024
            gpus_ram_free_gb = gpus_ram_free_gb + mem["free_memory"] / 1024
            cur_device.update(mem)
            devices[gpu_name] = cur_device
    return gpus_ram_total_gb, gpus_ram_free_gb, devices


def get_gpu_util_stats():
    ensure_installed("gputil")

    import GPUtil as GPU

    devices = dict()
    try:
        for gpu in GPU.getGPUs():
            cur_device = dict()
            for attr in "memoryTotal,memoryFree,load,driver,id,temperature,uuid".split(","):
                val = getattr(gpu, attr)
                if "memory" in attr:
                    val = val / 1024
                elif attr == "load":
                    val = val * 100
                cur_device[attr] = val
            devices[gpu.name] = cur_device
    except Exception as e:
        logger.exception(e)
    return devices


def get_pycuda_gpu_info():
    ensure_installed("pycuda")

    import pycuda.autoinit
    import pycuda.driver as cuda

    # free,total=cuda.mem_get_info()
    # print(free,total,"Global memory occupancy: %f%% free"%(free*100/total))

    devices = dict()
    try:
        for devicenum in range(cuda.Device.count()):
            device = cuda.Device(devicenum)
            cur_device = dict()
            cur_device["compute_capability"] = ".".join(map(str, device.compute_capability()))
            cur_device["total_memory"] = device.total_memory()
            attrs = device.get_attributes()
            features = dict()
            for (key, value) in attrs.items():
                features[str(key)] = value
            cur_device["features"] = features
            devices[device.name()] = cur_device
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
    ensure_installed("pylspci")

    try:
        devices = json.loads(check_output(["pylspci", "-nn"]).decode("utf-8"))
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
    return res


# ----------------------------------------------------------------------------------------------------------------------------
# Software info
# ----------------------------------------------------------------------------------------------------------------------------


def get_system_fingerprint(force_ml_stack=False):
    import platform, sys, importlib

    libs = "sklearn,keras,numpy,pandas,numba".split(",")
    vals = [sys.version, platform.machine(), platform.platform()]

    if force_ml_stack:
        for modulename in libs:
            tmp = importlib.import_module(modulename)

    text = "Python version: %s\nplatform: %s\nsystem: %s"
    for modulename in sys.modules:
        if modulename in libs:
            version = ""
            try:
                version = sys.modules[modulename].__version__
            except:
                pass
            if len(version) > 0:
                vals.append(version)
                text = text + "\n" + modulename + ": %s"
    return text % tuple(vals)


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
    from pympler import asizeof

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

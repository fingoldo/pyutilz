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
    exclude_pros: set = None,
    ensure_float: set = None,
    ensure_int: set = None,
    decode_dict: dict = None,
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
    if decode_dict is None:
        decode_dict = {}
    if ensure_float is None:
        ensure_float = set()
    if ensure_int is None:
        ensure_int = set()
    if exclude_pros is None:
        exclude_pros = set()
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
    exclude_pros: set = None,
    ensure_float: set = None,
    ensure_int: set = None,
    decode_dict: dict = None,
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
    if decode_dict is None:
        decode_dict = {}
    if ensure_float is None:
        ensure_float = set()
    if ensure_int is None:
        ensure_int = set()
    if exclude_pros is None:
        exclude_pros = set()
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

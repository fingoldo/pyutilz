"""Closes gaps in python itself.

Works on general matters such as types, reading settings, etc.
"""

# pylint: disable=wrong-import-order,wrong-import-position,unidiomatic-typecheck,pointless-string-statement

# *****************************************************************************************************************************************************
# IMPORTS
# *****************************************************************************************************************************************************

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# -----------------------------------------------------------------------------------------------------------------------------------------------------

from typing import *  # noqa: F401 pylint: disable=wildcard-import,unused-wildcard-import

import time
import numba
import numbers
import inspect
from datetime import datetime, date, timezone

from os.path import abspath, exists, join
import joblib
import errno
import os

import re
import inspect

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

import importlib, subprocess


def ensure_installed(packages, sep: str = " ") -> None:
    if packages:
        if type(packages) == str:
            if sep in packages:
                packages = packages.split(sep)
            else:
                packages = [packages]
        missing_packages = [pkg for pkg in packages if not importlib.util.find_spec(pkg)]
        if missing_packages:
            mes = f"Installing missing packages: {missing_packages}"
            logger.info(mes)
            for pkg in missing_packages:
                try:
                    subprocess.check_call(["pip", "install", pkg])
                except:
                    pass


# from .pythonlib import ensure_installed  # lint: disable=ungrouped-imports,disable=wrong-import-order

# ensure_installed("joblib")

# ----------------------------------------------------------------------------------------------------------------------------
# Operations on objects
# ----------------------------------------------------------------------------------------------------------------------------


def show_methods(obj, uppercased=False):
    return [a for a in dir(obj) if "__" not in a and (uppercased is False or a[0].isupper())]


# ----------------------------------------------------------------------------------------------------------------------------
# Operations on dicts
# ----------------------------------------------------------------------------------------------------------------------------


def prefix_dict_elems(obj: dict, prefix: str) -> dict:
    """Keys of dict assumed to be string"""
    return {(prefix + key): value for key, value in obj.items()}


def populate_object_from_dict(obj, dct):
    """Populates a class/object with properties from a dictionary."""
    for key in dct:
        setattr(obj, key, dct[key])


def flatten_keys_to_dict(obj: object) -> dict:
    """Recursively walks content of the object, bringing all the key-value keys to the top level dict."""
    res = dict()
    if type(obj) == list:
        for subobj in obj:
            res.update(flatten_keys_to_dict(subobj))
    elif type(obj) == dict:
        for key, value in obj.items():
            if type(value) in (list, dict):
                res.update(flatten_keys_to_dict(value))
            else:
                res[key] = value
    return res


def flatten_keys_to_set(
    obj: object,
    dict_merge_symbol: str = ":",
    stringify: bool = False,
    verbose: bool = False,
    max_chars: int = 10,
) -> dict:
    """Recursively walks content of an object, bringing all the key-value keys to the top level set."""
    res = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            # print(key,value)
            if isinstance(value, (dict, Iterable)):
                # print('pasing recursively %s' % key)
                res.update(flatten_keys_to_set(value, dict_merge_symbol=dict_merge_symbol))
            else:
                if dict_merge_symbol:
                    res.add(str(key) + dict_merge_symbol + str(value))
                else:
                    res.add(str(key))
                    res.add(str(value))
    elif isinstance(obj, str):
        res.add(obj)
    elif isinstance(obj, numbers.Number):
        if stringify:
            res.add(str(obj))
        else:
            res.add(obj)
    elif isinstance(obj, Iterable):
        for subobj in obj:
            res.update(flatten_keys_to_set(subobj, dict_merge_symbol=dict_merge_symbol))
    else:
        if stringify:
            if verbose:
                tmp_str = str(obj)
                logger.info(f"Processing object of type {type(obj)}, size {len(tmp_str):_} as a string.")
            res.add(str(obj))
        else:
            if verbose:
                tmp_str = str(obj)
                logger.info(f"Skipping object of type {type(obj)}, size {len(tmp_str):_}: {tmp_str[max_chars:]} .")
    return res


def ensure_dict_elem(obj: dict, name: str, value) -> None:
    """
    Make sure certain key exists in the dict
    """
    if name not in obj:
        obj[name] = value


def get_attr(obj: dict, attr_name: str, default_value: object = [], unwanted_value=None) -> object:
    """
    if attr is None, return default
    To prevent TypeError: 'NoneType' object is not iterable
    """
    if obj == unwanted_value:
        return default_value
    res = obj.get(attr_name, default_value)
    if res == unwanted_value:
        return default_value
    else:
        return res


def keys_changed_enough(obj: dict, prev_obj: dict, min_change_percent: float = 10.0, key_contains: str = None) -> bool:
    """Signals if numerical keys (optionally containing some value) of a dict-like object have changed by at least some percent.
    >>>keys_changed_enough(obj={"a": 100, "b": 180, "c": 300}, prev_obj={"a": 100, "b": 200, "c": 300}, min_change_percent=10.0, key_contains="b")
    True

    >>>keys_changed_enough(obj={"a": 100, "b": 181, "c": 300}, prev_obj={"a": 100, "b": 200, "c": 300}, min_change_percent=10.0, key_contains="b")
    False

    >>>keys_changed_enough(obj={"a": 100, "b": 220, "c": 300}, prev_obj={"a": 100, "b": 200, "c": 300}, min_change_percent=10.0, key_contains="b")
    True

    >>>keys_changed_enough(obj={"a": 100, "b": 221, "c": 300}, prev_obj={"a": 100, "b": 200, "c": 300}, min_change_percent=10.0, key_contains="b")
    True

    """
    for key, prev_value in prev_obj.items():
        if key_contains in key:
            if is_float(prev_value):
                new_value = obj.get(key)
                if is_float(new_value):
                    prev_value = to_float(prev_value)
                    if prev_value != 0.0:
                        new_value = to_float(new_value)
                        change = abs(new_value - prev_value) * 100 / prev_value
                        if change >= min_change_percent:
                            return True
                    else:
                        return True

    return False


# ----------------------------------------------------------------------------------------------------------------------------
# Operations on sequences
# ----------------------------------------------------------------------------------------------------------------------------


def unpack_counter(cntr: list) -> list:
    """
    Makes plain list of tokens out of Counter() result (which is a list of tuples:
    [('surgery', 252),('operating_room', 251),('operating_theatre', 251),)...
    """
    res = []
    for item in cntr:
        res.append(item[0])
    return res


def ensure_list_set_tuple(obj):
    if type(obj).__name__ in ("list", "set", "frozenset", "tuple`"):
        return obj
    else:
        return [obj]


def anyof_elements_in_string(elems: Sequence, target: str) -> bool:
    """
    Check if any of list elements are part of target string
    """
    res = False
    if elems:
        for sign in elems:
            if sign in target:
                res = True
                break
    return res


def filter_elements_by_type(obj: Union[dict, Sequence], allowed_types: tuple = (numbers.Number, str)) -> Union[dict, Sequence]:
    """
    Only leaves
    >>>filter_elems_by_type(obj=dict(a="test", b=3), allowed_types=(str))
    {'a': 'test'}

    >>>filter_elems_by_type(obj={"a", 1, "test"}, allowed_types=(str,))
    ['test', 'a']
    """
    if type(obj) == dict:
        return {key: value for key, value in obj.items() if isinstance(value, allowed_types)}
    else:
        return [value for value in obj if isinstance(value, allowed_types)]


def batch(iterable, n=1):
    """
    Batches a sequence

    >for x in batch(list(range(0, 10)), 3):print(x)
    [0, 1, 2]
    [3, 4, 5]
    [6, 7, 8]
    [9]
    """
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


# ----------------------------------------------------------------------------------------------------------------------------
# Sorting
# ----------------------------------------------------------------------------------------------------------------------------


def list_is_non_increasing(l: Iterable) -> bool:
    """Checks if a list is sorted."""
    return all(l[i] >= l[i + 1] for i in range(len(l) - 1))


def list_is_non_decreasing(l: Iterable) -> bool:
    """Checks if a list is sorted."""
    return all(l[i] <= l[i + 1] for i in range(len(l) - 1))


def sort_dict_by_value(dct: dict, reverse: bool = False) -> dict:
    return {k: v for k, v in sorted(dct.items(), key=lambda item: item[1], reverse=reverse)}


def sort_dict_by_key(dct: dict, reverse: bool = False) -> dict:
    return dict(sorted(dct.items()), reverse=reverse)


# ----------------------------------------------------------------------------------------------------------------------------
# Numerics
# ----------------------------------------------------------------------------------------------------------------------------


def is_float(string):
    try:
        float(str(string).replace(",", ""))
        return True
    except ValueError:
        return False


def to_float(string):
    return float(str(string).replace(",", ""))


# ----------------------------------------------------------------------------------------------------------------------------
# Time & dates
# ----------------------------------------------------------------------------------------------------------------------------


def utc_to_local(utc_dt):
    return utc_dt.replace(tzinfo=timezone.utc).astimezone(tz=None)


def utc_ts_2_locstr(
    inp_dt: str,
    dst: int = None,
    dst_names: dict = {
        1: "сек",
        60: "мин",
        60 * 60: "ч",
        60 * 60 * 24: "дн",
        60 * 60 * 24 * 30: "мес",
        60 * 60 * 24 * 365: "гд",
    },
    input_date_format="%Y-%m-%dT%H:%M:%S.%f",
    output_date_format="%Y-%m-%d %H:%M:%S",
) -> str:
    """
    Format UTC timestamp & optionally calculate time passed

    >utc_ts_2_locstr(inp_dt="2021-09-22T15:14:34.532707", dst=60)
    '2021-09-22 18:14:34 (102 мин. тому назад)'

    """
    if inp_dt is None or inp_dt == "":
        return ""
    utc_dt = datetime.strptime(inp_dt, input_date_format)
    res = f"{utc_to_local(utc_dt).strftime(output_date_format)}"

    if dst:
        if type(dst) == int:
            if dst > 0:
                if dst in dst_names:
                    utc_now = datetime.utcnow()
                    res += f" ({(utc_now - utc_dt).total_seconds() // dst:.0f} {dst_names.get(dst)}. тому назад)"
    return res


def read_timezoned_ts(inp):
    """
    read_timezoned_ts('2020-02-20T11:54:00.000-07:00')->'2020-02-20T11:54:00.000-0700'
    """
    parts = None
    for token in ("+", "-"):
        if token in inp:
            parts = inp.split(token)
            break
    if parts is None:
        return inp
    else:
        if len(parts) >= 2:
            return token.join((token.join(parts[:-1]), parts[-1].replace(":", "")))
        else:
            logger.error("Unexpected: split by %s of ts %s returned less than 2 results" % (token, inp))
            return inp


def datetime_to_unix_ts(dt):
    from datetime import datetime

    return int((dt - datetime(1970, 1, 1)).total_seconds())


def get_utc_unix_ts_seconds() -> int:
    return int(time.time())


def get_utc_unix_ts_milliseconds() -> int:
    return int(time.time() * 1000)


def imitate_delay(
    min_delay_seconds: float,
    max_delay_seconds: float,
    last_call_ts: Optional[object] = None,
    b_force: Optional[bool] = True,
    big_delay_prob: Optional[float] = None,
    big_delay_multiplier: Optional[float] = 10,
) -> object:
    """
    Waits random time interval (delay) since the last action.
    >>>last_call_ts=None;last_call_ts=imitate_delay(2,4,last_call_ts);last_call_ts=imitate_delay(2,4,last_call_ts);

    """
    from datetime import datetime
    from random import uniform, random
    from time import sleep

    if min_delay_seconds >= max_delay_seconds:
        logger.warning(f"min_delay_seconds of {min_delay_seconds} >= max_delay_seconds of {max_delay_seconds}!")
        max_delay_seconds = min_delay_seconds * 2

    if last_call_ts or b_force:
        random_delay = uniform(min_delay_seconds, max_delay_seconds)
        if big_delay_prob:
            if big_delay_prob > 0:
                if random() < big_delay_prob:
                    random_delay = random_delay * big_delay_multiplier
        if b_force and (last_call_ts is None):
            cur_delay = 0
        else:
            cur_delay = (datetime.utcnow() - last_call_ts).total_seconds()
        if cur_delay < random_delay:
            logger.debug("Sleeping %.2f sec." % (random_delay - cur_delay))
            sleep(random_delay - cur_delay)
    return datetime.utcnow()


def weekofmonth(date: date):
    return date.day // 7 + 1


def datetime_to_utc_timestamp(dt):
    """Converts a datetime object to UTC timestamp

    naive datetime will be considered UTC.

    """
    import calendar

    return calendar.timegm(dt.utctimetuple())


def age(dob: date) -> int:
    today = date.today()
    years = today.year - dob.year
    if today.month < dob.month or (today.month == dob.month and today.day < dob.day):
        years -= 1
    return years


# ----------------------------------------------------------------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------------------------------------------------------------


def get_or_warn(obj: dict, field: str, target: str) -> Optional[Any]:
    desired = obj.get(field)
    if desired is None:
        logger.warning(f"No {field} field in {target} {obj}")
    return desired


class CustomException(Exception):
    def __init__(self, code, message):
        self.code = code
        self.message = message

    def __str__(self):
        return repr("Error #" + str(self.code) + ": " + self.message)


cSTRUCTURED_FORMAT_PARSING_ERROR = 100


# ----------------------------------------------------------------------------------------------------------------------------
# Stack
# ----------------------------------------------------------------------------------------------------------------------------


def lookup_in_stack(variable):

    st = inspect.stack()
    for i in range(len(st)):
        frame = st[i]
        caller_globals = dict(inspect.getmembers(frame[0]))["f_globals"]
        res = caller_globals.get(variable)
        if res:
            return res


def get_parent_func_args(skip_args: Sequence = ("self",)) -> dict:
    """Get arg-values of a caller func as a dict."""

    previous_frame = inspect.currentframe().f_back
    args_info = inspect.getargvalues(previous_frame)

    # Collecting args-values of my_func in a dictionary
    argvals = {arg: args_info.locals.get(arg) for arg in args_info.args if arg not in skip_args}
    return argvals


def store_params_in_object(obj: object, params: dict, postfix: str = ""):
    """Useful for persisting __init__ params in the class instance."""
    if obj is None:
        return
    for param_name, param_value in params.items():
        setattr(obj, param_name + postfix, param_value)


def load_object_params_into_func(obj: object, locals: dict, postfix: str = "_param_"):
    """Contrary action to store_params_in_object, but does not work with locals (())."""
    if obj is None:
        return
    for attr in dir(obj):
        if attr.endswith(postfix):
            key, value = attr[: -len(postfix)], getattr(obj, attr)
            locals[key] = value


# ----------------------------------------------------------------------------------------------------------------------------
# Filesystem
# ----------------------------------------------------------------------------------------------------------------------------


def get_partitioned_filepath(fname: str, depth: int = 2, replace_char="_"):
    """Returns optimized filepath derived from the first N (depth) chars of the filename, to view them manually in a more optimized form
    Instead of an enormously huge chunk of files in a single dir.
    Chars are lowercased and replaced with a safe symbol if needed, to stcik to the names allowed by a filesystem.
    """
    from os import sep

    folders = []
    for char in fname[:depth].lower():
        if char.isalnum():
            folders.append(char)
        else:
            folders.append(replace_char)
    return sep.join(folders) + sep


def ensure_valid_filename(name: str, max_length: int = 255) -> str:
    """
    Replace invalid characters on Linux/Windows/MacOS with underscores.
    List from https://stackoverflow.com/a/31976060/819417
    Trailing spaces & periods are ignored on Windows.
    >>> fix_filename("  COM1  ")
    '_ COM1 _'
    >>> fix_filename("COM10")
    'COM10'
    >>> fix_filename("COM1,")
    'COM1,'
    >>> fix_filename("COM1.txt")
    '_.txt'
    >>> all('_' == fix_filename(chr(i)) for i in list(range(32)))
    True
    """
    return re.sub(
        r'[/\\:|<>"?*\0-\x1f]|^(AUX|COM[1-9]|CON|LPT[1-9]|NUL|PRN)(?![^.])|^\s|[\s.]$',
        "_",
        name[:max_length],
        flags=re.IGNORECASE,
    )


def load_file(fpath: str, unpickle_to_pd: bool = True, **kwargs):
    """
    Load plicked object, dataframe, Catboost model, based on file presence and name.
    """
    from catboost import CatBoostClassifier
    import pandas as pd, numpy as np

    is_here = False
    fpath = abspath(fpath)
    try:
        is_here = exists(fpath)
    except:
        pass
    if not is_here:
        logger.warning(f"File {fpath} not found!")
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), fpath)
        return None
    else:
        if fpath.lower().endswith(".joblib"):
            return joblib.load(fpath)
        elif fpath.lower().endswith(".pckl"):
            if unpickle_to_pd:
                return pd.read_pickle(fpath)
        elif fpath.lower().endswith(".bin"):
            # if "catboost" in fpath.lower():
            clf = CatBoostClassifier()
            return clf.load_model(fpath)


class ObjectsAndFilesProcessor:
    """Container objects processing via interacting with the filesystem."""

    def process_objects(
        self,
        objects_names: Union[str, Iterable] = None,
        container: Optional[dict] = None,
        names_sep: str = " ",
        path: str = "",
        namespace: str = "",
        namespace_sep: str = "_",
        file_extension: str = ".pckl",
        verbose: bool = True,
    ) -> int:
        """Walks over all сontainer's (dict-like) objects assosiated with files (one file per object), applies subclass-specific processing.

        If container is not specified, globals of the calling module are used.
        Objects names must be given exactly.

        Args:
            objects_names: list of object names to be populated. If it's a string, names_sep parameter is used to split it.
            If empty, all keys are processed.
            container: dictionary to be populated with objects names as keys, objects as values.
            names_sep: objects names separator.
            path: relative or absolute directory where the files are located.
            namespace: optional prefix added to files names.
            namespace_sep: separator of the namespace and file name.
            file_extension: default files extension to look for.
            verbose: whether to show or hide warnings in the log.

        Returns:
            number of succesfully processed files.
        """
        nprocessed = 0
        fpath = abspath(path)

        # Split names if needed
        if isinstance(objects_names, str):
            objects_names = objects_names.split(sep=names_sep)

        # Get caller globals is no container specified
        if container is None:
            caller_globals = dict(inspect.getmembers(inspect.stack()[1][0]))["f_globals"]
            container = caller_globals

        if not objects_names:
            objects_names = container.keys()

        for obj_name in objects_names:

            file_name = f"{obj_name}{file_extension}"
            if namespace:
                file_name = namespace + namespace_sep + file_name
            file_name = join(fpath, file_name)

            if self._process_object(container=container, obj_name=obj_name, file_name=file_name, verbose=verbose):
                nprocessed += 1
            else:
                if verbose:
                    logger.warning(f"Skipped object {obj_name}.")
        return nprocessed

    def _process_object(self, container: dict, obj_name: str, file_name: str, verbose: bool = True):
        # This method should be overridden in the subclasses
        raise NotImplementedError


class ObjectsDumper(ObjectsAndFilesProcessor):
    """Dumps container to disk.

    Usage Example:
        >>>ObjectsDumper().process_objects("discovered_fields required_arguments nested_fields", path="vars")
        3
    """

    def __init__(self, process_fcn: Callable = joblib.dump, process_kwargs: dict = {"compress": 9}, rewrite_existing: bool = True):
        self.process_fcn = process_fcn
        self.process_kwargs = process_kwargs
        self.rewrite_existing = rewrite_existing

    def _process_object(self, container: dict, obj_name: str, file_name: str, verbose: bool = True):
        # Do not rewrite existing non-empty objects/keys, warn instead.
        obj = container.get(obj_name)
        if obj:
            if self.rewrite_existing or not exists(file_name):
                self.process_fcn(obj, file_name, **self.process_kwargs)
                return True


class ObjectsLoader(ObjectsAndFilesProcessor):
    """Populates container from disk.

    Usage Example:
        >>>ObjectsLoader(rewrite_existing=True).process_objects("discovered_fields required_arguments nested_fields", path="vars")
        3
    """

    def __init__(self, process_fcn: Callable = joblib.load, process_kwargs: dict = {}, rewrite_existing: bool = False):
        self.process_fcn = process_fcn
        self.process_kwargs = process_kwargs
        self.rewrite_existing = rewrite_existing

    def _process_object(self, container: dict, obj_name: str, file_name: str, verbose: bool = True):
        if exists(file_name):
            if not self.rewrite_existing:
                # Do not rewrite existing non-empty objects/keys, warn instead.
                obj = container.get(obj_name)
                proceed = obj is None or (isinstance(obj, Iterable) and len(obj) == 0)
            else:
                proceed = True

                if proceed:
                    container[obj_name] = self.process_fcn(file_name, **self.process_kwargs)
                    return True


def get_human_readable_set_size(set_size: int, rounding: int = 1) -> str:
    """Converts integer number of records into something human-redable

    >>>get_human_readable_set_size(100500)
    '100.5K'

    """
    for power, letter in ((9, "B"), (6, "M"), (3, "K")):
        if set_size >= 10**power:
            return str(round(set_size / 10**power, rounding)) + letter
    return str(round(set_size, rounding))


class hashabledict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))

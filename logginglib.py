# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from .pythonlib import ensure_installed

ensure_installed("pywin32 concurrent-log-handler")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

from datetime import datetime, date
from os.path import sep, basename
from random import random
import numbers, inspect
import functools

import pdb
import logging
from logging import Handler, Logger
from logging.handlers import RotatingFileHandler  # QueueHandler, TimedRotatingFileHandler

# cd C:\ProgramData\Anaconda3\Scripts
# python pywin32_postinstall.py -install

from .pythonlib import filter_elements_by_type, ensure_dict_elem
from .strings import json_pg_dumps, suffixize
from .system import get_script_file
from .db import safe_execute_values
from .web import get_external_ip

EXTERNAL_IP = None
logger: Logger = None


def init_logging(
    custom_logger: object = None,
    level=logging.INFO,
    default_caller_name: str = "app.py",
    format: str = (
        "%(asctime)s - %(levelname)s - %(processName)s[%(process)d]-%(threadName)s[%(thread)d]"
        "-%(filename)s-%(name)s - %(funcName)s-line:%(lineno)d - %(message)s"
    ),
    log_to_file: bool = True,
    maxBytes: int = 5_000_000,
    forced_filename: str = None,
):
    global EXTERNAL_IP

    # EXTERNAL_IP = get_external_ip()

    if forced_filename:
        caller_name = forced_filename
    else:
        caller_name = inspect.stack()[1].filename.split(sep)[-1]  # get_script_file(__file__)
        if "ipython-input-" in caller_name or "ipykernel" in caller_name:
            caller_name = default_caller_name

    handlers = []
    if log_to_file:
        try:
            from concurrent_log_handler import ConcurrentRotatingFileHandler
        except Exception as e:
            handlers.append(RotatingFileHandler(filename=caller_name.replace(".py", ".log"), backupCount=2, mode="a", encoding="utf-8", maxBytes=maxBytes))
        else:
            handlers.append(
                ConcurrentRotatingFileHandler(
                    filename=caller_name.replace(".py", ".log"), backupCount=2, mode="a", encoding="utf-8", maxBytes=maxBytes, use_gzip=True
                )
            )

    handlers.append(logging.StreamHandler())

    logging.basicConfig(
        handlers=handlers,
        format=format,
        level=level,
    )

    global logger

    if custom_logger is None:
        logger = logging.getLogger(caller_name)
    else:
        logger = custom_logger
    return logger


def initialize_function_log(explicit_only: bool = False, allowed_types: tuple = (numbers.Number, str)) -> dict:
    """
    Used to log execution of the parent function.
    Creates a "starting frame", where (non-default) arguments of the parent function are saved in a dict,
    and starting timestamp is added.
    """

    function_name = None
    module_name = None

    current_frame = inspect.currentframe()

    try:
        current_frame = current_frame.f_back
        function_name = current_frame.f_code.co_name
        module_name = inspect.getfile(current_frame)
    except Exception as e:
        logging.exception(e)
    try:
        params = inspect.getargvalues(current_frame).locals
        kwargs = params.get("kwargs", {})
        kwargs = filter_elements_by_type(obj=kwargs, allowed_types=allowed_types)
        if explicit_only:
            params = kwargs
        else:
            params = filter_elements_by_type(obj=params, allowed_types=allowed_types)
            params = {**params, **kwargs}
    except Exception as e:
        logging.exception(e)

    results_log = {}
    results_log["module"] = basename(module_name)
    results_log["function"] = function_name
    results_log["parameters"] = params

    results_log["results"] = {"timing": {}}

    _init_clocks(results_log["results"]["timing"])

    return results_log


def _init_clocks(obj: dict) -> None:
    obj["started_at"] = datetime.utcnow()


def _stop_clocks(obj: dict) -> float:
    finished_at = datetime.utcnow()
    obj["finished_at"] = finished_at

    duration = (finished_at - obj.get("started_at")).total_seconds()
    if duration < 0:
        duration = 0
    obj["duration"] = duration

    return duration


def _message(activity_name: str):
    if activity_name:
        logger.info(activity_name + ("" if activity_name.strip()[-1] in "!.?," else "..."))


def _close_opened_activities(activities: dict) -> float:
    # let's close unclosed acts. returns duration of last closed activity.
    last_activity_duration = None
    for _, activity in activities.items():
        if not activity.get("finished_at"):
            last_activity_duration = _stop_clocks(activity)
    return last_activity_duration


def finalize_function_log(results_log: dict, db_path: str = None, verbose: bool = False) -> dict:
    """
    db_path: shema.table. example: logging.api_calls
    fields: list of fields. example: module, function, params, results
    """
    if "activities" in results_log["results"]:
        _close_opened_activities(results_log["results"]["activities"])

    _stop_clocks(results_log["results"]["timing"])

    if db_path:
        db_fields = "module,function,parameters,results,node,session"
        safe_execute_values(
            f"insert into {db_path} ({db_fields}) values %s",
            [[field if type(field) != dict else json_pg_dumps(field) for field in [results_log.get(field) for field in db_fields.split(",")]]],
        )
    else:
        if verbose:
            print(results_log)
    return results_log


def log_result(results_log: dict, key: str, value, verbose: bool = True) -> None:
    results_log["results"][key] = value
    if verbose:
        _message(f"{key}: {value}")


def log_results(results_log: dict, results: dict, verbose: bool = True) -> None:
    for key, value in results.items():
        results_log["results"][key] = value
    if verbose:
        _message(f"{results}")


def log_activity(results_log: dict, activity_name: str, verbose: bool = True) -> float:
    ensure_dict_elem(obj=results_log["results"], name="activities", value={})
    activities = results_log["results"]["activities"]
    last_activity_duration = _close_opened_activities(activities)

    if activity_name:
        ensure_dict_elem(obj=activities, name=activity_name, value={})
        activities[activity_name]["started_at"] = datetime.utcnow()

        if verbose:
            _message(activity_name)

    return last_activity_duration


def log_loaded_rows(obj: object, source: str, source_type: str = "db_table", results_log: dict = {}, lang: str = None, verbose: bool = False):
    assert source_type in ("db_table", "file")
    if lang is None:
        lang = globals().get("reports_language", "en")

    sources = {
        "db_table": {"en": "DB table", "ru": "таблицы БД"},
        "file": {"en": "file", "ru": "файла"},
    }

    if lang == "en":
        message = f"Loaded {len(obj):_} {suffixize('row',len(obj))} from {source} {sources.get(source_type,{}).get(lang,'')}."
    elif lang == "ru":
        message = f"Загружено {len(obj):_} строк из {sources.get(source_type,{}).get(lang,'')} {source}."

    ensure_dict_elem(obj=results_log["results"], name="loaded", value={})
    ensure_dict_elem(obj=results_log["results"]["loaded"], name=source_type, value={})

    results_log["results"]["loaded"][source_type][source] = {"rows": len(obj)}

    if verbose:
        _message(message)


def logged(db_path: str = None, explicit_only: bool = False, allowed_types: tuple = (numbers.Number, str), include_node_ip: bool = True):
    def decorator_logged(func):
        @functools.wraps(func)
        def wrapper_logged(*args, **kwargs):
            special_vars = ("current_proxy", "current_proxy_resolved", "login")

            function_name = func.__name__
            current_frame = inspect.currentframe().f_back
            module_name = inspect.getfile(current_frame)

            f_kwargs = filter_elements_by_type(obj=kwargs, allowed_types=allowed_types)
            if explicit_only:
                params = f_kwargs
            else:
                f_params = filter_elements_by_type(obj={i: val for i, val in enumerate(args)}, allowed_types=allowed_types)
                params = {**f_params, **f_kwargs}

            if "results_log" not in kwargs:
                kwargs["results_log"] = {}

            results_log = kwargs["results_log"]

            results_log["module"] = basename(module_name)
            results_log["function"] = function_name
            results_log["parameters"] = {key: value for key, value in params.items() if key not in special_vars}

            if include_node_ip:
                results_log["node"] = {"ip": EXTERNAL_IP}

            for var in special_vars:
                if var in kwargs:
                    if kwargs[var]:
                        results_log["session"] = {}
                        break
            for var in special_vars:
                if var in kwargs:
                    if kwargs[var]:
                        results_log["session"][var] = kwargs[var]

            results_log["results"] = {"timing": {}}
            _init_clocks(results_log["results"]["timing"])

            value = func(*args, **kwargs)
            finalize_function_log(kwargs["results_log"], db_path=db_path)
            return value

        return wrapper_logged

    return decorator_logged


class RedisHandler(Handler):
    """
    Logs into redis
    """

    def __init__(
        self,
        rc: object,
        LOG_DEST: Optional[str] = "Logs",
        LOG_SIZE: Optional[int] = 1000,
    ) -> None:
        Handler.__init__(self)
        self.rc = rc
        self.LOG_DEST = LOG_DEST
        self.LOG_SIZE = LOG_SIZE

    def emit(self, record):
        msg = self.format(record)
        try:
            self.rc.lpush(self.LOG_DEST, msg)
            if random() < 0.1:
                self.rc.ltrim(self.LOG_DEST, 0, self.LOG_SIZE)
                print("logging list trimmed!")
        except:
            pass


def debugged():
    def decorator_debugged(func):
        @functools.wraps(func)
        def wrapper_debugged(*args, **kwargs):
            done = False
            while not done:
                try:
                    value = func(*args, **kwargs)
                    done = True
                except Exception as e:
                    logger.exception(e)
                    pdb.set_trace()
            return value

        return wrapper_debugged

    return decorator_debugged

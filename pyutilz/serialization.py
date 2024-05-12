# ----------------------------------------------------------------------------------------------------------------------------
# Everything related to persisting objects into memory or disk!
# ----------------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging
logger=logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from .python import ensure_installed
ensure_installed("")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

import sys
import pickle, zlib, os
from . import system


def str_to_class(classname):
    pos1 = classname.find("(")
    init_data = ""
    if pos1 > 0:
        pos2 = classname.find(")", pos1)
        if pos2 > 0:
            init_data = classname[pos1:pos2]
            classname = classname[:pos1]
    opts = classname.split(".")
    if len(opts) > 1:
        the_module = ".".join(opts[:-1])
        the_class = opts[-1]
        return getattr(sys.modules[the_module], the_class)(init_data)
    else:
        return getattr(sys.modules[__name__], classname)(init_data)


def serialize(obj, fname: str = None, compression: int = 9):
    """
        If fname is passed, data will be persisted onto disk and success code will be returned
        Otherwise, serialized representation of the obj in memory will be returned.
    """
    if not (compression is None):
        assert isinstance(compression, int)
        assert compression >= -1 and compression <= 9
    try:
        data = pickle.dumps(obj)
        if not (compression is None):
            data = zlib.compress(data, compression)
        if not (fname is None):
            if type(fname) == str:
                systemutils.ensure_dir_exists(fname)
                with open(fname, "wb") as f:
                    f.write(data)
            elif type(fname).__name__ == "_io.BufferedWriter":
                f.write(data)
            return True
        else:
            return data
    except Exception as e:
        logger.exception(e)


def unserialize(obj, compression: int = 9):
    """
        If fname is passed, data will be read from disk.
        Otherwise, obj will be read from memory directl.
        Unpacked data will be returned.
    """
    if not (compression is None):
        assert isinstance(compression, int)
        assert compression >= -1 and compression <= 9
    try:
        to = type(obj).__name__
        if to == "str":
            if not os.path.isfile(obj):
                logger.error("File %s not found" % obj)
                return
            else:
                with open(obj, "rb") as f:
                    obj = f.read()
        elif to == "_io.BufferedReader":
            obj = f.read()

        if type(obj) == bytes:
            if not (compression is None):
                try:
                    data = zlib.decompress(obj)
                except Exception as e:
                    if "incorrect data check" in str(e):
                        logger.warn("Data seems to be not compressed")
                    else:
                        logger.exception(e)
                        return
            data = pickle.loads(data)
            return data
        else:
            logger.warning("Unexpected input data type: %s" % type(obj))
    except Exception as e:
        logger.exception(e)
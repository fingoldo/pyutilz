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

from pyutilz.core.pythonlib import ensure_installed
ensure_installed("")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import Any, Optional, Union

import sys
import io
import pickle, zlib, os
from pyutilz.system import system


def str_to_class(classname: str) -> Any:
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


def serialize(obj: Any, fname: Optional[Union[str, io.IOBase]] = None, compression: Optional[int] = 9) -> Union[bool, bytes, None]:
    """
        If fname is passed, data will be persisted onto disk and success code will be returned
        Otherwise, serialized representation of the obj in memory will be returned.
    """
    if compression is not None:
        assert isinstance(compression, int)
        assert compression >= -1 and compression <= 9
    try:
        data = pickle.dumps(obj)
        if compression is not None:
            data = zlib.compress(data, compression)
        if fname is not None:
            if isinstance(fname, str):
                system.ensure_dir_exists(fname)
                with open(fname, "wb") as f:
                    f.write(data)
            elif isinstance(fname, io.IOBase):
                fname.write(data)
            else:
                raise TypeError(f"Unsupported fname type for serialize: {type(fname)}")
            return True
        else:
            return data
    except Exception as e:
        logger.exception(e)


def unserialize(obj: Union[str, bytes, io.IOBase], compression: Optional[int] = 9) -> Any:
    """
        If fname is passed, data will be read from disk.
        Otherwise, obj will be read from memory directl.
        Unpacked data will be returned.
    """
    if compression is not None:
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
        elif isinstance(obj, io.IOBase):
            obj = obj.read()

        if isinstance(obj, bytes):
            data = obj
            if compression is not None:
                try:
                    data = zlib.decompress(obj)
                except zlib.error:
                    # zlib raises "incorrect header check" / "incorrect data check" for uncompressed input:
                    # fall back to treating obj as raw (uncompressed) pickle bytes.
                    logger.warning("Data seems to be not compressed; reading as raw pickle")
                    data = obj
            data = pickle.loads(data)
            return data
        else:
            logger.warning("Unexpected input data type: %s" % type(obj))
    except Exception as e:
        logger.exception(e)

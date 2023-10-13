# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from .pythonlib import ensure_installed

# ensure_installed("")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *


import glob
from os.path import join, getsize, sep

import PIL.Image
from PIL.ExifTags import TAGS
from PIL.TiffImagePlugin import IFDRational
from PIL import ImageFile

from sys import getsizeof

from .strings import remove_json_attributes, leave_json_attributes, json_pg_dumps, remove_json_defaults, extract_json_attribute


def ensure_bytes_converted(obj: dict) -> dict:
    """Ensure image information (as per Pillow) is converted from bytes to strings."""
    for field, value in obj.copy().items():
        if type(value) == bytes:
            try:
                obj[field] = value.decode("UTF-8")
            except:
                del obj[field]
        elif type(value) == dict:
            ensure_bytes_converted(value)
        elif type(value) in (tuple, IFDRational):
            obj[field] = str(value)
    return obj


def get_image_properties(img, skip_empty_exif: bool = True, filesize: int = None) -> dict:
    """
    For an image, uising PIL, read dimensions, info, exif data into a JSON-serializable dict
    """

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    orig_img = img

    # filesize

    if type(img) == str:
        filesize = getsize(img)
        try:
            img = PIL.Image.open(img)
        except Exception as e:
            logger.exception(e)
            return None, None

    # get exif

    try:
        exifdata = img.getexif()
    except Exception as e:
        logger.exception(e)
        exifdata = {}

    # decode exif

    decoded_exif = {}
    for tag_id, tag_value in exifdata.items():
        if type(tag_value) in (tuple, IFDRational):
            tag_value = str(tag_value)
        # get the tag name, instead of human unreadable tag id
        tag = TAGS.get(tag_id)
        if tag:
            data = exifdata.get(tag_id)
            # decode bytes
            if isinstance(tag_value, bytes):
                try:
                    tag_value = tag_value.decode("UTF-8")
                except:
                    logger.warning(f"Error when decoding byte property {tag}: {tag_value} in image {orig_img if type(orig_img)==str else ''}")
                    continue

            decoded_exif[tag] = tag_value
        else:
            if not skip_empty_exif:
                decoded_exif[tag_id] = tag_value

    width, height = img.size

    info = img.info

    if info:
        ensure_bytes_converted(info)

        remove_json_defaults(info, {"jfif": 257, "jfif_version": "(1, 1)", "jfif_density": "(1, 1)"}, warn_if_not_default=False)

        if "exif" in info:
            if not decoded_exif:
                decoded_exif = info["exif"].copy()
            del info["exif"]

    # convert to bytes

    try:
        image_bytes = img.tobytes()
    except Exception as e:
        raise (e)

    # construct response object

    if width == 0 and height == 0:
        res = {"height": height, "width": width, "size": 0}
    else:
        res = {"height": height, "width": width, "size": getsizeof(image_bytes)}

    if filesize:
        res["filesize"] = filesize

    if decoded_exif:
        res["exif"] = decoded_exif

    if info:
        res["info"] = info

    return image_bytes, res

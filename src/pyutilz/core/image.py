"""Image metadata utilities: EXIF/info extraction and JSON-safe normalization of Pillow-derived values."""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------


# ensure_installed("")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------


from os.path import getsize

import PIL.Image
from PIL.ExifTags import TAGS
from PIL.TiffImagePlugin import IFDRational
from PIL import ImageFile

from sys import getsizeof

from pyutilz.text.strings import remove_json_defaults
from typing import Optional, Tuple


def ensure_bytes_converted(obj: dict) -> dict:
    """Ensure image information (as per Pillow) is converted from bytes to strings."""
    for field, value in obj.copy().items():
        if isinstance(value, bytes):
            try:
                obj[field] = value.decode("UTF-8")
            except Exception:
                logger.warning("Error when decoding byte property %s: %r", field, value)
                del obj[field]
        elif isinstance(value, dict):
            ensure_bytes_converted(value)
        elif type(value) in (tuple, IFDRational):
            obj[field] = str(value)
    return obj


def get_image_properties(
    img,
    skip_unknown_exif_tags: bool = True,
    filesize: Optional[int] = None,
    allow_truncated: bool = False,
    skip_empty_exif: Optional[bool] = None,
) -> Tuple[Optional[bytes], Optional[dict]]:
    """
    For an image, uising PIL, read dimensions, info, exif data into a JSON-serializable dict.

    ``skip_unknown_exif_tags`` (default True) drops exif entries whose tag id Pillow's ``TAGS``
    table cannot name. It has nothing to do with EMPTY exif -- its former name, ``skip_empty_exif``,
    described behaviour the function never had; turning it off injects raw integer keys and
    undecoded ``bytes`` values into a dict this function promises is JSON-serializable, so
    ``json.dumps`` then raises "keys must be str...". ``skip_empty_exif`` remains accepted as a
    deprecated alias.

    ``filesize`` is honoured when supplied by the caller (only looked up from disk when it was not),
    and a genuine 0 is reported rather than dropped.

    ``allow_truncated`` tolerates a truncated image for the duration of THIS call only. Pillow's
    ``ImageFile.LOAD_TRUNCATED_IMAGES`` is a process-global that changes decoding for every other
    PIL user in the process, so it is restored on exit rather than left flipped on.
    """
    if skip_empty_exif is not None:
        logger.warning("get_image_properties: `skip_empty_exif` is a deprecated alias of `skip_unknown_exif_tags`; please rename the argument.")
        skip_unknown_exif_tags = skip_empty_exif

    prev_load_truncated = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = allow_truncated

    orig_img = img

    # filesize

    opened_here = False
    if isinstance(img, str):
        if filesize is None:
            filesize = getsize(img)
        try:
            img = PIL.Image.open(img)
            opened_here = True
        except Exception as e:
            logger.exception(e)
            ImageFile.LOAD_TRUNCATED_IMAGES = prev_load_truncated
            return None, None

    try:
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
                exifdata.get(tag_id)
                # decode bytes
                if isinstance(tag_value, bytes):
                    try:
                        tag_value = tag_value.decode("UTF-8")
                    except Exception:
                        logger.warning(
                            "Error when decoding byte property %s: %s in image %s",
                            tag,
                            tag_value,
                            orig_img if isinstance(orig_img, str) else "",
                        )
                        continue

                decoded_exif[tag] = tag_value
            else:
                if not skip_unknown_exif_tags:
                    decoded_exif[tag_id] = tag_value

        width, height = img.size

        info = img.info

        if info:
            # Captured BEFORE normalization: ensure_bytes_converted either decodes this value to
            # str or deletes the key, so the fallback below could never see the original mapping.
            raw_exif = info.get("exif")
            ensure_bytes_converted(info)

            remove_json_defaults(info, {"jfif": 257, "jfif_version": "(1, 1)", "jfif_density": "(1, 1)"}, warn_if_not_default=False)

            if "exif" in info:
                if not decoded_exif and hasattr(raw_exif, "copy") and not isinstance(raw_exif, (bytes, str)):
                    decoded_exif = raw_exif.copy()
                del info["exif"]

        # convert to bytes

        image_bytes = img.tobytes()

        # construct response object

        if width == 0 and height == 0:
            res = {"height": height, "width": width, "size": 0}
        else:
            res = {"height": height, "width": width, "size": getsizeof(image_bytes)}

        if filesize is not None:
            res["filesize"] = filesize

        if decoded_exif:
            res["exif"] = decoded_exif

        if info:
            res["info"] = info

        return image_bytes, res
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = prev_load_truncated
        if opened_here:
            img.close()

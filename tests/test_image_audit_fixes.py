"""Regression tests for the 2026-09-02 audit -- pyutilz.core.image findings (real Pillow)."""

from __future__ import annotations

import io

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image, ImageFile  # noqa: E402

from pyutilz.core.image import get_image_properties  # noqa: E402


def _jpeg_bytes(exif: bytes | None = None) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (4, 4), (10, 20, 30))
    if exif is None:
        img.save(buf, format="JPEG")
    else:
        img.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


def _write(tmp_path, name, data):
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


def test_get_image_properties_restores_the_global_truncated_flag(tmp_path):
    """Reading metadata must not permanently reconfigure Pillow for the whole process."""
    path = _write(tmp_path, "plain.jpg", _jpeg_bytes())
    for initial in (False, True):
        ImageFile.LOAD_TRUNCATED_IMAGES = initial
        get_image_properties(path)
        assert ImageFile.LOAD_TRUNCATED_IMAGES is initial
    ImageFile.LOAD_TRUNCATED_IMAGES = False


def test_get_image_properties_restores_the_flag_on_an_unopenable_path(tmp_path):
    path = _write(tmp_path, "broken.jpg", b"not an image")
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    assert get_image_properties(path) == (None, None)
    assert ImageFile.LOAD_TRUNCATED_IMAGES is False


def test_get_image_properties_allow_truncated_is_opt_in(tmp_path):
    path = _write(tmp_path, "plain.jpg", _jpeg_bytes())
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    get_image_properties(path, allow_truncated=True)
    assert ImageFile.LOAD_TRUNCATED_IMAGES is False


def test_get_image_properties_survives_a_utf8_decodable_exif_blob(tmp_path):
    """The info["exif"] fallback used to call .copy() on a value already turned into a str."""
    path = _write(tmp_path, "exif.jpg", _jpeg_bytes(exif=b"Exif\x00\x00"))
    image_bytes, props = get_image_properties(path)
    assert image_bytes is not None
    assert props is not None
    assert "exif" not in props.get("info", {})

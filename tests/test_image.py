"""Tests for pyutilz.core.image — ensure_bytes_converted function."""

import pytest
import sys
from unittest.mock import MagicMock


class _FakeIFDRational:
    def __init__(self, val):
        self.val = val

    def __str__(self):
        return f"IFDRational({self.val})"


@pytest.fixture(autouse=True, scope="module")
def _stub_pil():
    """Temporarily stub PIL modules so image.py can be imported without Pillow.

    ``pyutilz.core.image`` binds ``IFDRational`` as a module-level name at import
    time (``from PIL.TiffImagePlugin import IFDRational``); once imported, that
    binding is cached in ``sys.modules`` for the rest of the process. If some
    earlier test (in another module, order randomized by pytest-randomly)
    already imported ``pyutilz.core.image`` against the REAL PIL, later imports
    here would silently reuse that stale binding instead of ``_FakeIFDRational``,
    making the isinstance-style check in ``ensure_bytes_converted`` never match.
    Drop ``pyutilz.core.image`` from ``sys.modules`` too, so ``_get_func()``'s
    import is always fresh against THIS module's stubs.
    """
    _tiff_mod = MagicMock()
    _tiff_mod.IFDRational = _FakeIFDRational

    stubs = {
        "PIL": MagicMock(),
        "PIL.Image": MagicMock(),
        "PIL.ExifTags": MagicMock(TAGS={}),
        "PIL.TiffImagePlugin": _tiff_mod,
        "PIL.ImageFile": MagicMock(),
    }

    saved = {k: sys.modules.get(k) for k in stubs}
    saved_image_mod = sys.modules.pop("pyutilz.core.image", None)
    sys.modules.update(stubs)
    yield
    # Restore originals
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v
    sys.modules.pop("pyutilz.core.image", None)
    if saved_image_mod is not None:
        sys.modules["pyutilz.core.image"] = saved_image_mod


def _get_func():
    from pyutilz.core.image import ensure_bytes_converted
    return ensure_bytes_converted


def test_bytes_decoded_to_string():
    assert _get_func()({"a": b"hello"}) == {"a": "hello"}


def test_invalid_bytes_deleted():
    assert "a" not in _get_func()({"a": b"\x80\x81\x82"})


def test_nested_dict_processed():
    assert _get_func()({"outer": {"inner": b"val"}}) == {"outer": {"inner": "val"}}


def test_tuple_converted_to_string():
    assert _get_func()({"t": (1, 2, 3)}) == {"t": "(1, 2, 3)"}


def test_ifdrational_converted_to_string():
    assert _get_func()({"r": _FakeIFDRational(42)}) == {"r": "IFDRational(42)"}


def test_mixed_types():
    result = _get_func()({
        "s": b"ok",
        "bad": b"\xff\xfe",
        "nested": {"x": b"y"},
        "tup": (1,),
        "num": 42,
        "txt": "hello",
    })
    assert result["s"] == "ok"
    assert "bad" not in result
    assert result["nested"] == {"x": "y"}
    assert result["tup"] == "(1,)"
    assert result["num"] == 42
    assert result["txt"] == "hello"


def test_empty_dict():
    assert _get_func()({}) == {}


def _get_properties_func():
    from pyutilz.core.image import get_image_properties
    return get_image_properties


def test_get_image_properties_closes_handle_on_success(tmp_path):
    """When given a path string, get_image_properties() must close the PIL handle it opened."""
    fake_img = MagicMock()
    fake_img.getexif.return_value = {}
    fake_img.size = (10, 10)
    fake_img.info = {}
    fake_img.tobytes.return_value = b"\x00" * 100

    import pyutilz.core.image as image_mod
    image_mod.PIL.Image.open.return_value = fake_img

    fpath = tmp_path / "fake.png"
    fpath.write_bytes(b"not a real image, just needs to exist for getsize()")

    get_image_properties = _get_properties_func()
    image_bytes, res = get_image_properties(str(fpath))

    assert image_bytes == b"\x00" * 100
    fake_img.close.assert_called_once()


def test_get_image_properties_closes_handle_on_exception(tmp_path):
    """The file handle must be closed even when processing (e.g. tobytes()) raises."""
    fake_img = MagicMock()
    fake_img.getexif.return_value = {}
    fake_img.size = (10, 10)
    fake_img.info = {}
    fake_img.tobytes.side_effect = OSError("corrupt image data")

    import pyutilz.core.image as image_mod
    image_mod.PIL.Image.open.return_value = fake_img

    fpath = tmp_path / "corrupt.png"
    fpath.write_bytes(b"corrupt bytes")

    get_image_properties = _get_properties_func()
    with pytest.raises(OSError):
        get_image_properties(str(fpath))

    fake_img.close.assert_called_once()


def test_get_image_properties_does_not_close_caller_provided_image():
    """If caller passes an already-open Image object, this function must not close it."""
    fake_img = MagicMock()
    fake_img.getexif.return_value = {}
    fake_img.size = (10, 10)
    fake_img.info = {}
    fake_img.tobytes.return_value = b"\x00" * 100

    get_image_properties = _get_properties_func()
    image_bytes, res = get_image_properties(fake_img)

    assert image_bytes == b"\x00" * 100
    fake_img.close.assert_not_called()


def test_invalid_bytes_logs_warning(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="pyutilz.core.image"):
        result = _get_func()({"a": b"\x80\x81\x82"})

    assert "a" not in result
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert "a" in caplog.records[0].message

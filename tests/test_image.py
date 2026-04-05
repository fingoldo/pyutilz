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
    """Temporarily stub PIL modules so image.py can be imported without Pillow."""
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
    sys.modules.update(stubs)
    yield
    # Restore originals
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


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

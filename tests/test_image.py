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

    import pyutilz.core as _pyutilz_core

    saved = {k: sys.modules.get(k) for k in stubs}
    saved_image_mod = sys.modules.pop("pyutilz.core.image", None)
    # sys.modules.pop() does NOT clear the submodule binding on the PARENT package, so the
    # re-import triggered inside this fixture rebinds ``pyutilz.core.image`` to the
    # stub-built module -- and restoring sys.modules alone never undoes that. Anything later
    # doing ``from pyutilz.core import image`` would then silently get the MagicMock-PIL
    # version, with behaviour depending on file order (order IS randomised here, see
    # requirements-dev.txt / pytest-randomly). Save and restore the attribute explicitly.
    _HAD_ATTR = hasattr(_pyutilz_core, "image")
    saved_image_attr = getattr(_pyutilz_core, "image", None)
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
    if _HAD_ATTR:
        _pyutilz_core.image = saved_image_attr
    else:
        # Nothing was bound before; drop whatever the stubbed re-import left behind.
        if hasattr(_pyutilz_core, "image"):
            delattr(_pyutilz_core, "image")


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


def test_get_image_properties_closes_handle_on_success(tmp_path, monkeypatch):
    """When given a path string, get_image_properties() must close the PIL handle it opened."""
    fake_img = MagicMock()
    fake_img.getexif.return_value = {}
    fake_img.size = (10, 10)
    fake_img.info = {}
    fake_img.tobytes.return_value = b"\x00" * 100

    import pyutilz.core.image as image_mod

    pil_stub = MagicMock()
    pil_stub.Image.open.return_value = fake_img
    monkeypatch.setattr(image_mod, "PIL", pil_stub)

    fpath = tmp_path / "fake.png"
    fpath.write_bytes(b"not a real image, just needs to exist for getsize()")

    get_image_properties = _get_properties_func()
    image_bytes, res = get_image_properties(str(fpath))

    assert image_bytes == b"\x00" * 100
    fake_img.close.assert_called_once()


def test_get_image_properties_closes_handle_on_exception(tmp_path, monkeypatch):
    """The file handle must be closed even when processing (e.g. tobytes()) raises."""
    fake_img = MagicMock()
    fake_img.getexif.return_value = {}
    fake_img.size = (10, 10)
    fake_img.info = {}
    fake_img.tobytes.side_effect = OSError("corrupt image data")

    import pyutilz.core.image as image_mod

    pil_stub = MagicMock()
    pil_stub.Image.open.return_value = fake_img
    monkeypatch.setattr(image_mod, "PIL", pil_stub)

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


# ----------------------------------------------------------------------------------------------------------------------------
# Behaviour with real Pillow images: dimensions, EXIF decoding, the deprecated alias and the failure paths.
# ----------------------------------------------------------------------------------------------------------------------------

import logging as _logging
import os as _os

from pyutilz.core import image as image_mod

_MAKE_TAG = 0x010F  # "Make" in Pillow's TAGS table
_UNKNOWN_TAG = 0x9999


def _save_jpeg(tmp_path, name="a.jpg", size=(7, 5), make=None, unknown=None):
    from PIL import Image

    exif = None
    if make is not None or unknown is not None:
        exif = Image.Exif()
        if make is not None:
            exif[_MAKE_TAG] = make
        if unknown is not None:
            exif[_UNKNOWN_TAG] = unknown
    path = tmp_path / name
    kwargs = {"exif": exif.tobytes()} if exif is not None else {}
    Image.new("RGB", size, (10, 20, 30)).save(path, "JPEG", **kwargs)
    return path


class _FakeImage:
    """Stands in for an already-opened image: the attributes ``get_image_properties`` reads, no file involved."""

    def __init__(self, size=(0, 0), info=None, exif=None, data=b""):
        self.size = size
        self.info = {} if info is None else info
        self._exif = {} if exif is None else exif
        self._data = data

    def getexif(self):
        return self._exif

    def tobytes(self):
        return self._data


class TestImageProperties:
    @pytest.fixture(autouse=True)
    def _real_pillow(self, monkeypatch):
        # The module-wide fixture swaps Pillow for stubs in sys.modules; these tests need the real decoder, so put
        # the real package back (and re-bind image_mod to it) for their duration, then restore the stubs.
        import importlib

        stubbed = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "PIL" or k.startswith("PIL.")}
        monkeypatch.setitem(sys.modules, "pyutilz.core.image", image_mod)
        try:
            importlib.import_module("PIL.Image")
            importlib.reload(image_mod)
            yield
        finally:
            for k in [k for k in sys.modules if k == "PIL" or k.startswith("PIL.")]:
                sys.modules.pop(k)
            sys.modules.update(stubbed)

    def test_the_dimensions_and_pixel_bytes_come_back_in_order(self, tmp_path):
        path = _save_jpeg(tmp_path, size=(7, 5))
        image_bytes, res = image_mod.get_image_properties(str(path))
        assert res["width"] == 7
        assert res["height"] == 5
        assert len(image_bytes) == 7 * 5 * 3

    def test_the_size_is_the_byte_length_of_the_pixels(self, tmp_path):
        path = _save_jpeg(tmp_path, size=(7, 5))
        image_bytes, res = image_mod.get_image_properties(str(path))
        assert res["size"] == _getsizeof(image_bytes)

    def test_the_file_size_is_read_from_disk_when_not_supplied(self, tmp_path):
        path = _save_jpeg(tmp_path)
        _, res = image_mod.get_image_properties(str(path))
        assert res["filesize"] == _os.path.getsize(path)

    def test_a_supplied_file_size_is_used_even_when_zero(self, tmp_path):
        path = _save_jpeg(tmp_path)
        _, res = image_mod.get_image_properties(str(path), filesize=0)
        assert res["filesize"] == 0

    def test_a_make_tag_is_decoded_to_its_name(self, tmp_path):
        path = _save_jpeg(tmp_path, make="Acme")
        _, res = image_mod.get_image_properties(str(path))
        assert res["exif"]["Make"] == "Acme"

    def test_an_unknown_tag_is_dropped_by_default_and_kept_when_asked(self, tmp_path):
        path = _save_jpeg(tmp_path, make="Acme", unknown="x")
        _, dropped = image_mod.get_image_properties(str(path))
        assert "Make" in dropped["exif"]
        assert _UNKNOWN_TAG not in dropped["exif"]
        _, kept = image_mod.get_image_properties(str(path), skip_unknown_exif_tags=False)
        assert _UNKNOWN_TAG in kept["exif"]

    def test_the_deprecated_alias_is_warned_about_and_applied(self, tmp_path, caplog):
        path = _save_jpeg(tmp_path, unknown="x")
        with caplog.at_level(_logging.WARNING):
            _, res = image_mod.get_image_properties(str(path), skip_empty_exif=False)
        assert _UNKNOWN_TAG in res["exif"]
        assert "get_image_properties: `skip_empty_exif` is a deprecated alias of `skip_unknown_exif_tags`; please rename the argument." in [
            r.getMessage() for r in caplog.records
        ]

    def test_a_non_image_file_gives_none_and_logs_the_error(self, tmp_path, caplog):
        path = tmp_path / "notes.txt"
        path.write_text("not an image")
        with caplog.at_level(_logging.ERROR):
            assert image_mod.get_image_properties(str(path)) == (None, None)
        assert caplog.records

    def test_the_truncation_flag_is_restored_after_the_call(self, tmp_path):
        from PIL import ImageFile

        before = ImageFile.LOAD_TRUNCATED_IMAGES
        path = _save_jpeg(tmp_path)
        image_mod.get_image_properties(str(path), allow_truncated=True)
        assert ImageFile.LOAD_TRUNCATED_IMAGES == before

    def test_a_zero_by_zero_image_reports_size_zero(self):
        image_bytes, res = image_mod.get_image_properties(_FakeImage(size=(0, 0), data=b""))
        assert res == {"height": 0, "width": 0, "size": 0}
        assert image_bytes == b""

    def test_a_missing_pillow_raises_with_the_install_hint(self, monkeypatch, tmp_path):
        monkeypatch.setattr(image_mod, "PIL", None)
        with pytest.raises(ImportError, match="get_image_properties needs Pillow; install the imaging extra"):
            image_mod.get_image_properties(_FakeImage())

    def test_unknown_tags_do_not_end_the_decoding(self):
        exif = {_UNKNOWN_TAG: "x", _MAKE_TAG: "Acme"}
        _, res = image_mod.get_image_properties(_FakeImage(size=(1, 1), exif=exif, data=b"\x00\x00\x00"))
        assert res["exif"] == {"Make": "Acme"}

    def test_a_bytes_value_is_decoded_to_text(self):
        exif = {_MAKE_TAG: b"Acme"}
        _, res = image_mod.get_image_properties(_FakeImage(size=(1, 1), exif=exif, data=b"\x00\x00\x00"))
        assert res["exif"]["Make"] == "Acme"

    def test_an_undecodable_bytes_value_is_skipped_with_a_warning_naming_it(self, caplog):
        exif = {_MAKE_TAG: b"\xff\xfe", _UNKNOWN_TAG: "x"}
        with caplog.at_level(_logging.WARNING):
            _, res = image_mod.get_image_properties(_FakeImage(size=(1, 1), exif=exif, data=b"\x00\x00\x00"))
        assert "Make" not in res.get("exif", {})
        messages = [r.getMessage() for r in caplog.records]
        assert any(m.startswith("Error when decoding byte property Make: ") for m in messages)

    def test_a_tuple_value_is_kept_as_its_text(self):
        exif = {_MAKE_TAG: (1, 2)}
        _, res = image_mod.get_image_properties(_FakeImage(size=(1, 1), exif=exif, data=b"\x00\x00\x00"))
        assert res["exif"]["Make"] == "(1, 2)"

    def test_the_info_exif_block_is_removed_from_the_info_mapping(self):
        info = {"exif": b"\x00", "dpi": (72, 72)}
        _, res = image_mod.get_image_properties(_FakeImage(size=(1, 1), info=info, data=b"\x00\x00\x00"))
        assert "exif" not in res.get("info", {})
        assert res["info"]["dpi"] == "(72, 72)"

    def test_the_default_jfif_fields_are_stripped_from_info(self):
        info = {"jfif": 257, "jfif_version": "(1, 1)", "jfif_density": "(1, 1)", "dpi": (72, 72)}
        _, res = image_mod.get_image_properties(_FakeImage(size=(1, 1), info=info, data=b"\x00\x00\x00"))
        assert "jfif" not in res["info"]
        assert "jfif_version" not in res["info"]


@pytest.fixture
def _real_pillow_fixture(monkeypatch):
    """Put the real Pillow back in place of the module-wide stub for one test, then restore the stub."""
    import importlib

    stubbed = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "PIL" or k.startswith("PIL.")}
    monkeypatch.setitem(sys.modules, "pyutilz.core.image", image_mod)
    try:
        importlib.import_module("PIL.Image")
        importlib.reload(image_mod)
        yield
    finally:
        for k in [k for k in sys.modules if k == "PIL" or k.startswith("PIL.")]:
            sys.modules.pop(k)
        sys.modules.update(stubbed)


@pytest.mark.usefixtures("_real_pillow_fixture")
class TestImageSecondRound:
    def test_an_undecodable_bytes_warning_names_the_field_then_its_value(self, caplog):
        with caplog.at_level(_logging.WARNING):
            image_mod.ensure_bytes_converted({"a": b"\x80\x81"})
        assert [r.getMessage() for r in caplog.records] == ["Error when decoding byte property a: b'\\x80\\x81'"]

    def test_an_undecodable_exif_bytes_warning_names_the_tag_the_value_and_the_image(self, caplog):
        with caplog.at_level(_logging.WARNING):
            image_mod._decode_exif({_MAKE_TAG: b"\xff\xfe"}, True, "img.jpg")
        assert [r.getMessage() for r in caplog.records] == ["Error when decoding byte property Make: b'\\xff\\xfe' in image img.jpg"]

    def test_an_undecodable_exif_value_from_a_non_path_image_names_no_image(self, caplog):
        with caplog.at_level(_logging.WARNING):
            image_mod._decode_exif({_MAKE_TAG: b"\xff"}, True, object())
        assert next(r.getMessage() for r in caplog.records).endswith(" in image ")

    def test_an_undecodable_tag_does_not_end_the_decoding(self):
        decoded = image_mod._decode_exif({_MAKE_TAG: b"\xff", 0x010E: "desc"}, True, "x")
        assert "Make" not in decoded
        assert decoded["ImageDescription"] == "desc"

    def test_the_exif_fallback_copies_the_info_block_when_nothing_was_decoded(self):
        info = {"exif": {"Make": "Acme"}}
        _, res = image_mod.get_image_properties(_FakeImage(size=(1, 1), info=info, data=b"\x00\x00\x00"))
        assert res["exif"] == {"Make": "Acme"}
        assert "exif" not in res.get("info", {})

    def test_decoded_exif_wins_over_the_info_block(self):
        info = {"exif": {"Other": "x"}}
        exif = {_MAKE_TAG: "Acme"}
        _, res = image_mod.get_image_properties(_FakeImage(size=(1, 1), info=info, exif=exif, data=b"\x00\x00\x00"))
        assert res["exif"] == {"Make": "Acme"}

    @pytest.mark.parametrize("raw", [b"\x00", "text", 7])
    def test_a_bytes_text_or_scalar_info_block_is_not_copied(self, raw):
        info = {"exif": raw}
        _, res = image_mod.get_image_properties(_FakeImage(size=(1, 1), info=info, data=b"\x00\x00\x00"))
        assert "exif" not in res
        assert "exif" not in res.get("info", {})

    def test_the_jfif_density_default_is_stripped_from_info(self):
        info = {"jfif_density": "(1, 1)", "dpi": (72, 72)}
        _, res = image_mod.get_image_properties(_FakeImage(size=(1, 1), info=info, data=b"\x00\x00\x00"))
        assert "jfif_density" not in res["info"]

    def test_a_non_default_jfif_value_is_kept_without_a_warning(self, caplog):
        info = {"jfif": 999, "dpi": (72, 72)}
        with caplog.at_level(_logging.WARNING):
            _, res = image_mod.get_image_properties(_FakeImage(size=(1, 1), info=info, data=b"\x00\x00\x00"))
        assert res["info"]["jfif"] == 999
        assert caplog.records == []

    def test_a_zero_width_alone_keeps_the_pixel_size(self):
        _, res = image_mod.get_image_properties(_FakeImage(size=(0, 5), data=b""))
        assert res["size"] == _getsizeof(b"")

    def test_a_getexif_failure_is_logged_and_gives_no_exif(self, caplog):
        class _Broken(_FakeImage):
            def getexif(self):
                raise ValueError("bad exif")

        with caplog.at_level(_logging.ERROR):
            _, res = image_mod.get_image_properties(_Broken(size=(1, 1), data=b"\x00\x00\x00"))
        assert "exif" not in res
        assert [r.getMessage() for r in caplog.records] == ["bad exif"]

    def test_truncation_is_refused_unless_asked_for(self, tmp_path):
        from PIL import Image

        path = tmp_path / "big.png"
        Image.new("RGB", (64, 64), (1, 2, 3)).save(path, "PNG")
        data = path.read_bytes()
        bad = tmp_path / "bad.png"
        bad.write_bytes(data[: len(data) // 2])
        with pytest.raises(OSError):
            image_mod.get_image_properties(str(bad))
        image_bytes, res = image_mod.get_image_properties(str(bad), allow_truncated=True)
        assert res["width"] == 64


def _getsizeof(obj):
    from sys import getsizeof

    return getsizeof(obj)

from __future__ import annotations
import base64
import logging
import tempfile
from pathlib import Path
import pytest
from pyutilz.llm._messages import images_on_disk


def _png_uri(media_type: str = "image/png") -> str:
    return f"data:{media_type};base64," + base64.b64encode(b"\x89PNG-bytes").decode()


class TestImagesOnDiskTraversal:
    def test_traversal_media_type_writes_nothing_outside_the_temp_dir(self, caplog):
        probe = Path(tempfile.gettempdir()) / "pyutilz_core1_trav_probe"
        probe.unlink(missing_ok=True)
        evil = "data:image/x/../../pyutilz_core1_trav_probe;base64," + base64.b64encode(b"hi").decode()
        with caplog.at_level(logging.WARNING), images_on_disk([evil]) as (fragment, written):
            assert fragment == ""
            assert written == []
        assert not probe.exists()
        assert "1 of 1 image(s) not sent" in caplog.text

    def test_allowlisted_types_get_fixed_extensions_and_are_removed_after(self):
        with images_on_disk([_png_uri("image/png"), _png_uri("image/jpeg"), _png_uri("image/svg+xml")]) as (fragment, written):
            assert [p.name for p in written] == ["attachment_1.png", "attachment_2.jpg", "attachment_3.svg"]
            directory = written[0].parent
            assert len(written) == 3 and all(p.parent == directory and p.read_bytes() == b"\x89PNG-bytes" for p in written)
            assert str(written[0]) in fragment
        assert not directory.exists()

    def test_unknown_image_subtype_is_dropped_not_written(self):
        with images_on_disk([_png_uri("image/x-evil"), _png_uri("image/png")]) as (_fragment, written):
            assert [p.name for p in written] == ["attachment_2.png"]

    def test_directory_removed_when_the_body_raises(self):
        seen: list[Path] = []
        with pytest.raises(RuntimeError), images_on_disk([_png_uri()]) as (_fragment, written):
            seen.extend(written)
            raise RuntimeError("boom")
        assert seen and not seen[0].parent.exists()

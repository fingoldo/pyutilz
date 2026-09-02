"""Tests for ``pyutilz.database.deltalakes``.

``is_local_path`` is public, is used by ``safe_delta_write`` (deltalakes.py:72) to decide whether
a file lock can be taken at all, and was previously never mentioned anywhere under tests/
(audit F20, 2026-09-02) -- a misclassified remote URI silently takes the local-lock path.
"""

import os

import pytest

from pyutilz.database.deltalakes import is_local_path


@pytest.mark.parametrize(
    "path",
    [
        "/var/data/table",
        "relative/table",
        "table",
        "file:///var/data/table",
        "",
    ],
)
def test_local_paths_are_local(path):
    assert is_local_path(path) is True


@pytest.mark.parametrize("path", ["s3://bucket/table", "azure://container/table"])
def test_object_store_uris_are_not_local(path):
    assert is_local_path(path) is False


@pytest.mark.parametrize("path", ["gs://bucket/table", "https://example.com/table", "abfss://c@a.dfs.core.windows.net/t"])
def test_other_remote_schemes_are_not_local(path):
    """Any recognised URI scheme that is not "" / "file" is remote -- the function does not need
    an allow-list entry per cloud vendor to refuse the local-lock path."""
    assert is_local_path(path) is False


@pytest.mark.skipif(os.name != "nt", reason="Windows drive-letter parsing is Windows-only behaviour")
def test_windows_drive_letter_is_local_not_a_uri_scheme():
    """``urlparse("R:/data")`` reports scheme "r" -- without the single-letter special case a
    perfectly ordinary Windows path would be classified as a remote store."""
    assert is_local_path(r"R:\data	able") is True
    assert is_local_path("C:/data/table") is True

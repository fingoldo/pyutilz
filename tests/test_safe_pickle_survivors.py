"""POSIX file-mode pin for safe_pickle.safe_dump (mutation survivor)."""

from __future__ import annotations

import os
import stat

import pytest

from pyutilz.core import safe_pickle


@pytest.mark.skipif(os.name != "posix", reason="file mode bits are only enforced on POSIX")
def test_a_dumped_payload_is_readable_only_by_its_owner(tmp_path):
    path = str(tmp_path / "payload.pkl")
    safe_pickle.safe_dump({"k": 1}, path)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600

from __future__ import annotations
from pathlib import Path
import pytest


def test_s13_persistent_download_error_gives_up(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("boto3")
    from unittest.mock import MagicMock

    from pyutilz.cloud import cloud as cloud_mod

    monkeypatch.setattr(cloud_mod, "S3_BUCKET_NAME", "mybucket")
    sleep_calls: list = []
    monkeypatch.setattr(cloud_mod, "sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr(cloud_mod, "s3_file_exists", lambda key, bucket: True)
    fake_s3 = MagicMock()
    fake_s3.meta.client.download_file.side_effect = OSError("access denied")
    monkeypatch.setattr(cloud_mod, "s3", fake_s3)
    monkeypatch.setattr(cloud_mod, "exists", lambda path: False)

    with pytest.raises(RuntimeError, match="Giving up"):
        cloud_mod.get_from_s3_or_cache(str(tmp_path / "model.bin"), "model.bin", str(tmp_path))
    assert fake_s3.meta.client.download_file.call_count == cloud_mod._MAX_DOWNLOAD_ATTEMPTS
    assert len(sleep_calls) == cloud_mod._MAX_DOWNLOAD_ATTEMPTS - 1

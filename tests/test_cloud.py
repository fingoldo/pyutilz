"""Regression tests for pyutilz.cloud.cloud (2026-07-21 audit fixes)."""

import os
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("boto3")

from pyutilz.cloud import cloud as cloud_mod


class TestConnectToS3CredentialFix:
    """Regression test: connect_to_s3() previously read config values into globals() instead
    of the local aws_access_key_id/aws_secret_access_key variables, so boto3.Session() was
    always called with (None, None) regardless of settings.ini's actual content."""

    def test_credentials_actually_reach_boto_session(self, tmp_path):
        ini_path = tmp_path / "settings.ini"
        ini_path.write_text("[S3]\naws_access_key_id = AKIAEXAMPLE\naws_secret_access_key = supersecret\n")

        captured = {}

        class _FakeSession:
            def __init__(self, aws_access_key_id=None, aws_secret_access_key=None):
                captured["aws_access_key_id"] = aws_access_key_id
                captured["aws_secret_access_key"] = aws_secret_access_key

            def resource(self, name):
                return MagicMock()

        with patch("boto3.Session", _FakeSession):
            cloud_mod.connect_to_s3(file=str(ini_path))

        assert captured["aws_access_key_id"] == "AKIAEXAMPLE"
        assert captured["aws_secret_access_key"] == "supersecret"  # pragma: allowlist secret

    def test_missing_section_passes_none_credentials_not_a_crash(self, tmp_path):
        """read_config_file() returns True even when the requested section/file has nothing to
        read (each missing variable is individually caught and set to None) -- connect_to_s3()
        must not crash in that case; it legitimately falls back to boto3's own ambient
        credential resolution (env vars / ~/.aws/credentials / IAM role) with (None, None)."""
        captured = {}

        class _FakeSession:
            def __init__(self, aws_access_key_id=None, aws_secret_access_key=None):
                captured["aws_access_key_id"] = aws_access_key_id
                captured["aws_secret_access_key"] = aws_secret_access_key

            def resource(self, name):
                return MagicMock()

        with patch("boto3.Session", _FakeSession):
            cloud_mod.connect_to_s3(file=str(tmp_path / "does_not_exist.ini"))

        assert captured["aws_access_key_id"] is None
        assert captured["aws_secret_access_key"] is None


class TestS3FileExists:
    def test_raises_clear_error_when_s3_not_connected(self):
        cloud_mod.s3 = None
        with pytest.raises(RuntimeError, match="connect_to_s3"):
            cloud_mod.s3_file_exists("key", "bucket")

    def test_404_returns_false(self):
        from botocore.exceptions import ClientError

        fake_s3 = MagicMock()
        fake_s3.meta.client.head_object.side_effect = ClientError({"Error": {"Code": "404"}}, "HeadObject")
        cloud_mod.s3 = fake_s3
        assert cloud_mod.s3_file_exists("key", "bucket") is False

    def test_found_returns_true(self):
        fake_s3 = MagicMock()
        fake_s3.meta.client.head_object.return_value = {}
        cloud_mod.s3 = fake_s3
        assert cloud_mod.s3_file_exists("key", "bucket") is True

    def test_auth_error_propagates_not_swallowed_as_false(self):
        """Regression test: a real auth/network ClientError (not a 404) must not be silently
        reported as "object doesn't exist"."""
        from botocore.exceptions import ClientError

        fake_s3 = MagicMock()
        fake_s3.meta.client.head_object.side_effect = ClientError({"Error": {"Code": "AccessDenied"}}, "HeadObject")
        cloud_mod.s3 = fake_s3
        with pytest.raises(ClientError):
            cloud_mod.s3_file_exists("key", "bucket")


class TestGetFromS3OrCacheCorruptZip:
    def test_corrupted_zip_is_removed_and_loop_sleeps_instead_of_spinning(self, tmp_path, monkeypatch):
        """Regression test: previously neither removed the corrupted zip nor slept on an unpack
        failure, spinning the while-loop at 100% CPU forever on the same unusable file."""
        monkeypatch.setattr(cloud_mod, "S3_BUCKET_NAME", "mybucket")
        sleep_calls = []
        monkeypatch.setattr(cloud_mod, "sleep", lambda s: sleep_calls.append(s))

        local_path = str(tmp_path / "model")
        zip_path = local_path + ".zip"
        with open(zip_path, "wb") as f:
            f.write(b"not a real zip file")

        # exists(local_object_path) is False the first time (enters the loop), True the second
        # (loop exits) -- exactly one iteration's worth of the corrupted-zip branch.
        # exists(zip_path) reflects the real filesystem, so it correctly flips to False once
        # the fix removes the corrupted file.
        exists_results = iter([False, True])

        def fake_exists(path):
            if path == zip_path:
                return os.path.exists(zip_path)
            return next(exists_results)

        monkeypatch.setattr(cloud_mod, "exists", fake_exists)

        with patch("shutil.unpack_archive", side_effect=ValueError("corrupted archive")):
            cloud_mod.get_from_s3_or_cache(local_path, "model.zip", str(tmp_path))

        assert not os.path.exists(zip_path), "corrupted zip must be removed so the next iteration re-downloads"
        assert sleep_calls == [10], "must sleep instead of busy-looping at zero delay"


class TestGetFromS3OrCacheDownloadFailure:
    def test_download_exception_sleeps_instead_of_spinning(self, tmp_path, monkeypatch):
        """Regression test: a persistent per-object download failure (bad IAM perms, always-
        corrupt remote object) previously retried with only network-RTT pacing -- no backoff --
        unlike this function's two sibling failure branches ("not found in bucket" and "corrupt
        zip", both sleep(10))."""
        monkeypatch.setattr(cloud_mod, "S3_BUCKET_NAME", "mybucket")
        sleep_calls = []
        monkeypatch.setattr(cloud_mod, "sleep", lambda s: sleep_calls.append(s))
        monkeypatch.setattr(cloud_mod, "s3_file_exists", lambda key, bucket: True)

        fake_s3 = MagicMock()
        fake_s3.meta.client.download_file.side_effect = OSError("connection reset")
        monkeypatch.setattr(cloud_mod, "s3", fake_s3)

        local_path = str(tmp_path / "model.bin")
        exists_results = iter([False, True])
        monkeypatch.setattr(cloud_mod, "exists", lambda path: next(exists_results))

        cloud_mod.get_from_s3_or_cache(local_path, "model.bin", str(tmp_path))

        assert sleep_calls == [10], "must sleep instead of busy-looping at zero delay"

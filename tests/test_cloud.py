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


import logging as _logging


class TestGcpStorage:
    def test_upload_builds_the_bucket_from_its_name_and_key_file_in_order(self, monkeypatch):
        seen = {}
        blob = MagicMock()
        bucket = MagicMock()
        bucket.blob.return_value = blob
        monkeypatch.setattr(cloud_mod, "_gcp_storage_bucket", lambda name, acc: seen.setdefault("args", (name, acc)) and bucket)
        cloud_mod.gcp_storage_upload_blob("bkt", "/tmp/src.bin", "dest/blob.bin", "key.json")
        assert seen["args"] == ("bkt", "key.json")
        bucket.blob.assert_called_once_with("dest/blob.bin")
        blob.upload_from_filename.assert_called_once_with("/tmp/src.bin")

    def test_upload_logs_source_and_destination_at_info(self, monkeypatch, caplog):
        monkeypatch.setattr(cloud_mod, "_gcp_storage_bucket", lambda name, acc: MagicMock())
        with caplog.at_level(_logging.INFO, logger=cloud_mod.logger.name):
            cloud_mod.gcp_storage_upload_blob("bkt", "/tmp/src.bin", "dest/blob.bin", "key.json")
        assert [r.getMessage() for r in caplog.records] == ["File /tmp/src.bin uploaded to dest/blob.bin."]

    def test_download_builds_the_bucket_and_fetches_the_named_blob(self, monkeypatch, caplog):
        seen = []
        blob = MagicMock()
        bucket = MagicMock()
        bucket.blob.return_value = blob
        monkeypatch.setattr(cloud_mod, "_gcp_storage_bucket", lambda name, acc: seen.append((name, acc)) or bucket)
        with caplog.at_level(_logging.INFO, logger=cloud_mod.logger.name):
            cloud_mod.gcp_storage_download_blob("bkt", "src/blob.bin", "/tmp/out.bin", "key.json")
        assert seen == [("bkt", "key.json")]
        bucket.blob.assert_called_once_with("src/blob.bin")
        blob.download_to_filename.assert_called_once_with("/tmp/out.bin")
        assert [r.getMessage() for r in caplog.records] == ["Blob src/blob.bin downloaded to /tmp/out.bin."]


class TestConnectToS3Defaults:
    def test_the_default_config_file_is_settings_ini_in_the_working_directory(self, tmp_path, monkeypatch):
        (tmp_path / "settings.ini").write_text("[S3]\naws_access_key_id = AKIADEFAULT\naws_secret_access_key = secretdefault\n")
        monkeypatch.chdir(tmp_path)
        captured = {}

        class _FakeSession:
            def __init__(self, aws_access_key_id=None, aws_secret_access_key=None):
                captured["id"] = aws_access_key_id

            def resource(self, name):
                return MagicMock()

        with patch("boto3.Session", _FakeSession):
            cloud_mod.connect_to_s3()
        assert captured["id"] == "AKIADEFAULT"


class TestS3FileExistsCodes:
    def test_no_such_key_is_a_missing_object_not_an_error(self):
        from botocore.exceptions import ClientError

        fake_s3 = MagicMock()
        fake_s3.meta.client.head_object.side_effect = ClientError({"Error": {"Code": "NoSuchKey"}}, "HeadObject")
        cloud_mod.s3 = fake_s3
        assert cloud_mod.s3_file_exists("key", "bucket") is False

    def test_the_head_request_names_the_bucket_and_key_exactly(self):
        fake_s3 = MagicMock()
        fake_s3.meta.client.head_object.return_value = {}
        cloud_mod.s3 = fake_s3
        cloud_mod.s3_file_exists("the/key", "the-bucket")
        fake_s3.meta.client.head_object.assert_called_once_with(Bucket="the-bucket", Key="the/key")


class TestGetFromS3OrCacheRetryBudget:
    def _run(self, monkeypatch, tmp_path, outcomes):
        """Drive get_from_s3_or_cache with scripted download outcomes (True = success, Exception = failure).

        A success writes the local file, so the enclosing poll loop ends exactly as it does in production.
        """
        monkeypatch.setattr(cloud_mod, "S3_BUCKET_NAME", "mybucket")
        sleep_calls = []
        monkeypatch.setattr(cloud_mod, "sleep", lambda s: sleep_calls.append(s))
        attempts = []
        local = str(tmp_path / "model.bin")

        def _download(local_path, key, bucket):
            attempts.append(key)
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            if outcome == "noop":
                return
            with open(local_path, "wb") as f:
                f.write(b"ok")

        monkeypatch.setattr(cloud_mod, "_download_model_from_s3", _download)
        return sleep_calls, attempts, local

    def test_the_budget_is_exactly_five_consecutive_failures(self, monkeypatch, tmp_path):
        sleep_calls, attempts, local = self._run(monkeypatch, tmp_path, [OSError("x")] * 5)
        with pytest.raises(RuntimeError, match="Giving up"):
            cloud_mod.get_from_s3_or_cache(local, "model.bin", str(tmp_path))
        assert len(attempts) == 5
        assert sleep_calls == [10, 10, 10, 10]

    def test_the_give_up_message_names_the_object_bucket_and_count(self, monkeypatch, tmp_path):
        _, _, local = self._run(monkeypatch, tmp_path, [OSError("x")] * 5)
        with pytest.raises(RuntimeError) as info:
            cloud_mod.get_from_s3_or_cache(local, "model.bin", str(tmp_path))
        assert str(info.value) == "Giving up on 'model.bin' from bucket 'mybucket' after 5 failed downloads"
        assert isinstance(info.value.__cause__, OSError)

    def test_a_success_resets_the_failure_count_to_zero(self, monkeypatch, tmp_path):
        """Four failures, a success, then four more failures must not give up: the count restarts at zero."""
        # "noop" is a download that returns without producing the file, so the poll loop keeps going.
        outcomes = [OSError("a")] * 4 + ["noop"] + [OSError("b")] * 4 + [True]
        sleep_calls, attempts, local = self._run(monkeypatch, tmp_path, outcomes)
        cloud_mod.get_from_s3_or_cache(local, "model.bin", str(tmp_path))
        assert len(attempts) == 10
        assert len(sleep_calls) == 8


class TestDownloadModelFromS3:
    def _s3(self, monkeypatch, exists=True):
        monkeypatch.setattr(cloud_mod, "S3_BUCKET_NAME", "mybucket")
        monkeypatch.setattr(cloud_mod, "s3_file_exists", lambda key, bucket: exists)
        sleeps = []
        monkeypatch.setattr(cloud_mod, "sleep", lambda s: sleeps.append(s))
        fake = MagicMock()
        monkeypatch.setattr(cloud_mod, "s3", fake)
        return fake, sleeps

    def test_existence_is_checked_with_key_then_bucket(self, monkeypatch):
        seen = []
        monkeypatch.setattr(cloud_mod, "s3_file_exists", lambda key, bucket: seen.append((key, bucket)) or True)
        monkeypatch.setattr(cloud_mod, "S3_BUCKET_NAME", "mybucket")
        monkeypatch.setattr(cloud_mod, "s3", MagicMock())
        cloud_mod._download_model_from_s3("/cache/model.bin", "models/model.bin", "mybucket")
        assert seen == [("models/model.bin", "mybucket")]

    def test_a_plain_object_downloads_straight_to_the_local_path(self, monkeypatch):
        fake, _ = self._s3(monkeypatch)
        cloud_mod._download_model_from_s3("/cache/model.bin", "models/model.bin", "mybucket")
        fake.meta.client.download_file.assert_called_once_with(Bucket="mybucket", Key="models/model.bin", Filename="/cache/model.bin")

    def test_a_zip_object_downloads_to_the_local_path_with_the_zip_suffix(self, monkeypatch):
        fake, _ = self._s3(monkeypatch)
        cloud_mod._download_model_from_s3("/cache/model", "models/model.zip", "mybucket")
        fake.meta.client.download_file.assert_called_once_with(Bucket="mybucket", Key="models/model.zip", Filename="/cache/model.zip")

    def test_a_success_is_logged_at_info_with_key_and_bucket(self, monkeypatch, caplog):
        self._s3(monkeypatch)
        with caplog.at_level(_logging.INFO, logger=cloud_mod.logger.name):
            cloud_mod._download_model_from_s3("/cache/model.bin", "models/model.bin", "mybucket")
        assert [r.getMessage() for r in caplog.records] == ["Downloaded model models/model.bin from bucket mybucket"]

    def test_a_failed_download_is_logged_then_reraised(self, monkeypatch, caplog):
        fake, _ = self._s3(monkeypatch)
        fake.meta.client.download_file.side_effect = OSError("reset")
        with caplog.at_level(_logging.ERROR, logger=cloud_mod.logger.name):
            with pytest.raises(OSError, match="reset"):
                cloud_mod._download_model_from_s3("/cache/model.bin", "models/model.bin", "mybucket")
        assert [r.getMessage() for r in caplog.records] == ["Error while downloading model models/model.bin from bucket mybucket: reset"]

    def test_a_missing_object_warns_and_pauses_before_the_next_poll(self, monkeypatch, caplog):
        fake, sleeps = self._s3(monkeypatch, exists=False)
        with caplog.at_level(_logging.WARNING, logger=cloud_mod.logger.name):
            cloud_mod._download_model_from_s3("/cache/model.bin", "models/model.bin", "mybucket")
        fake.meta.client.download_file.assert_not_called()
        assert sleeps == [10]
        assert [r.getMessage() for r in caplog.records] == ["Model models/model.bin not found in bucket mybucket"]


class TestUnpackLocalZip:
    def test_the_archive_is_unpacked_into_temp_dir_from_its_zip_path(self, tmp_path, monkeypatch):
        local = str(tmp_path / "model")
        open(local + ".zip", "wb").close()
        seen = []

        def _unpack(archive, dest):
            seen.append((archive, dest))
            open(local, "wb").close()

        monkeypatch.setattr("shutil.unpack_archive", _unpack)
        cloud_mod._unpack_local_zip(local, "models/model.zip", str(tmp_path))
        assert seen == [(local + ".zip", str(tmp_path))]
        assert not os.path.exists(local + ".zip")

    def test_a_successful_unpack_logs_the_archive_path(self, tmp_path, monkeypatch, caplog):
        local = str(tmp_path / "model")
        open(local + ".zip", "wb").close()

        def _unpack(archive, dest):
            open(local, "wb").close()

        monkeypatch.setattr("shutil.unpack_archive", _unpack)
        with caplog.at_level(_logging.INFO, logger=cloud_mod.logger.name):
            cloud_mod._unpack_local_zip(local, "models/model.zip", str(tmp_path))
        assert [r.getMessage() for r in caplog.records] == [f"Unzipped model from archive {local}.zip"]

    def test_an_unpack_that_does_not_produce_the_object_raises_with_the_layout_hint(self, tmp_path, monkeypatch):
        local = str(tmp_path / "model")
        open(local + ".zip", "wb").close()
        monkeypatch.setattr("shutil.unpack_archive", lambda archive, dest: None)
        monkeypatch.setattr(cloud_mod, "sleep", lambda s: None)
        expected = (
            f"get_from_s3_or_cache: archive models/model.zip was unpacked into {str(tmp_path)!r} "
            f"but {local!r} still does not exist -- check the archive layout "
            f"and that temp_dir is the directory the object is expected in"
        )
        with pytest.raises(FileNotFoundError) as info:
            cloud_mod._unpack_local_zip(local, "models/model.zip", str(tmp_path))
        assert str(info.value) == expected

    def test_a_corrupt_archive_is_logged_with_its_path_and_error(self, tmp_path, monkeypatch, caplog):
        local = str(tmp_path / "model")
        open(local + ".zip", "wb").close()
        monkeypatch.setattr(cloud_mod, "sleep", lambda s: None)

        def _boom(archive, dest):
            raise ValueError("bad crc")

        monkeypatch.setattr("shutil.unpack_archive", _boom)
        with caplog.at_level(_logging.ERROR, logger=cloud_mod.logger.name):
            cloud_mod._unpack_local_zip(local, "models/model.zip", str(tmp_path))
        assert [r.getMessage() for r in caplog.records] == [f"Error while unpacking model from bucket {local}.zip: bad crc"]

    def test_a_corrupt_archive_that_cannot_be_removed_is_logged_and_the_pause_still_happens(self, tmp_path, monkeypatch, caplog):
        local = str(tmp_path / "model")
        open(local + ".zip", "wb").close()
        sleeps = []
        monkeypatch.setattr(cloud_mod, "sleep", lambda s: sleeps.append(s))
        monkeypatch.setattr("shutil.unpack_archive", lambda archive, dest: (_ for _ in ()).throw(ValueError("bad")))

        def _no_remove(path):
            raise OSError("locked")

        monkeypatch.setattr(os, "remove", _no_remove)
        with caplog.at_level(_logging.ERROR, logger=cloud_mod.logger.name):
            cloud_mod._unpack_local_zip(local, "models/model.zip", str(tmp_path))
        assert sleeps == [10]
        assert f"Could not remove corrupted archive {local}.zip: locked" in [r.getMessage() for r in caplog.records]


class TestGetFromS3OrCacheEntry:
    def test_an_unconfigured_bucket_is_refused_with_the_configuration_hint(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cloud_mod, "S3_BUCKET_NAME", None)
        with pytest.raises(ValueError) as info:
            cloud_mod.get_from_s3_or_cache(str(tmp_path / "m"), "m", str(tmp_path))
        assert str(info.value) == "pyutilz.cloud.cloud.S3_BUCKET_NAME must be configured before calling get_from_s3_or_cache()"

    def test_a_local_zip_is_logged_and_unpacked_instead_of_downloaded(self, monkeypatch, tmp_path, caplog):
        monkeypatch.setattr(cloud_mod, "S3_BUCKET_NAME", "mybucket")
        local = str(tmp_path / "model")
        open(local + ".zip", "wb").close()
        downloads, unpacks = [], []
        monkeypatch.setattr(cloud_mod, "_download_model_from_s3", lambda *a: downloads.append(a))
        monkeypatch.setattr(cloud_mod, "_unpack_local_zip", lambda *a: unpacks.append(a) or open(local, "wb").close())
        with caplog.at_level(_logging.INFO, logger=cloud_mod.logger.name):
            cloud_mod.get_from_s3_or_cache(local, "models/model.zip", str(tmp_path))
        assert downloads == []
        assert unpacks == [(local, "models/model.zip", str(tmp_path))]
        assert [r.getMessage() for r in caplog.records] == ["Zipped version of models/model.zip found in local FS"]

    def test_a_download_passes_its_arguments_in_order(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cloud_mod, "S3_BUCKET_NAME", "mybucket")
        seen = []
        monkeypatch.setattr(cloud_mod, "_download_model_from_s3", lambda local, key, bucket: seen.append((local, key, bucket)) or open(local, "wb").close())
        local = str(tmp_path / "model.bin")
        cloud_mod.get_from_s3_or_cache(local, "models/model.bin", str(tmp_path))
        assert seen == [(local, "models/model.bin", "mybucket")]


class TestRemainingBranches:
    def test_the_resource_is_opened_for_the_s3_service(self, tmp_path, monkeypatch):
        (tmp_path / "settings.ini").write_text("[S3]\naws_access_key_id = A\naws_secret_access_key = B\n")
        services = []

        class _FakeSession:
            def __init__(self, aws_access_key_id=None, aws_secret_access_key=None):
                pass

            def resource(self, name):
                services.append(name)
                return MagicMock()

        with patch("boto3.Session", _FakeSession):
            cloud_mod.connect_to_s3(file=str(tmp_path / "settings.ini"))
        assert services == ["s3"]

    def test_a_plain_object_is_never_unpacked_even_when_a_zip_sits_beside_it(self, tmp_path, monkeypatch):
        local = str(tmp_path / "model")
        open(local + ".zip", "wb").close()
        unpacked = []
        monkeypatch.setattr("shutil.unpack_archive", lambda archive, dest: unpacked.append(archive))
        cloud_mod._unpack_local_zip(local, "models/model.bin", str(tmp_path))
        assert unpacked == []
        assert os.path.exists(local + ".zip")

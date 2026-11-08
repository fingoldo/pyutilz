"""Lock-directory, lock acquisition, timeout and logging pins for database.deltalakes (mutation survivors)."""

from __future__ import annotations

import inspect
import os
import types

import pytest

from pyutilz.database import deltalakes as dl


class _Log:
    def __init__(self):
        self.debug_calls: list = []
        self.warning_calls: list = []
        self.exception_calls: list = []

    def debug(self, *args):
        self.debug_calls.append(args)

    def warning(self, *args):
        self.warning_calls.append(args)

    def exception(self, *args):
        self.exception_calls.append(args)


@pytest.fixture
def log(monkeypatch):
    fake = _Log()
    monkeypatch.setattr(dl, "logger", fake)
    return fake


@pytest.fixture
def temp_root(tmp_path, monkeypatch):
    monkeypatch.setattr(dl.tempfile, "gettempdir", lambda: str(tmp_path))
    return tmp_path


class _FakeLock:
    created: list = []
    behaviour = "ok"

    def __init__(self, lock_file):
        self.lock_file = lock_file
        _FakeLock.created.append(lock_file)

    def acquire(self, timeout):
        if _FakeLock.behaviour == "timeout":
            raise dl.Timeout(self.lock_file)
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_lock(monkeypatch):
    _FakeLock.created = []
    _FakeLock.behaviour = "ok"
    monkeypatch.setattr(dl, "FileLock", _FakeLock)
    return _FakeLock


class TestLockDirectoryNaming:
    def test_the_user_name_from_the_environment_names_the_directory_on_windows(self, temp_root, monkeypatch):
        monkeypatch.delattr(dl.os, "getuid", raising=False)
        monkeypatch.setenv("USERNAME", "alice")
        base = dl._lock_dir()
        assert os.path.basename(base) == "pyutilz-delta-locks-alice"
        assert os.path.isdir(base)

    def test_a_missing_user_name_falls_back_to_user(self, temp_root, monkeypatch):
        monkeypatch.delattr(dl.os, "getuid", raising=False)
        monkeypatch.delenv("USERNAME", raising=False)
        assert os.path.basename(dl._lock_dir()) == "pyutilz-delta-locks-user"

    def test_the_directory_is_created_with_mode_0700(self, temp_root, monkeypatch):
        monkeypatch.delattr(dl.os, "getuid", raising=False)
        monkeypatch.setenv("USERNAME", "bob")
        calls: list = []
        real = os.makedirs
        monkeypatch.setattr(dl.os, "makedirs", lambda p, mode=0o777, exist_ok=False: calls.append((p, mode, exist_ok)) or real(p, exist_ok=exist_ok))
        dl._lock_dir()
        assert calls and calls[0][1] == 0o700 and calls[0][2] is True

    def test_a_symlinked_directory_is_refused(self, temp_root, monkeypatch):
        monkeypatch.delattr(dl.os, "getuid", raising=False)
        monkeypatch.setenv("USERNAME", "carol")
        monkeypatch.setattr(dl.os.path, "islink", lambda p: True)
        expected_base = os.path.join(str(temp_root), "pyutilz-delta-locks-carol")
        with pytest.raises(RuntimeError) as info:
            dl._lock_dir()
        assert str(info.value) == f"safe_delta_write: lock directory {expected_base!r} is a symlink; refusing to lock inside it"


class TestPosixOwnership:
    def test_the_user_id_names_the_directory_and_the_owner_check_passes(self, temp_root, monkeypatch):
        monkeypatch.setattr(dl.os, "getuid", lambda: 4242, raising=False)
        st = types.SimpleNamespace(st_uid=4242)
        monkeypatch.setattr(dl.os, "stat", lambda p: st)
        chmods: list = []
        monkeypatch.setattr(dl.os, "chmod", lambda p, m: chmods.append((p, m)))
        base = dl._lock_dir()
        assert os.path.basename(base) == "pyutilz-delta-locks-4242"
        assert chmods == [(base, 0o700)]

    def test_a_directory_owned_by_another_user_is_refused(self, temp_root, monkeypatch):
        monkeypatch.setattr(dl.os, "getuid", lambda: 4242, raising=False)
        monkeypatch.setattr(dl.os, "stat", lambda p: types.SimpleNamespace(st_uid=7))
        expected_base = os.path.join(str(temp_root), "pyutilz-delta-locks-4242")
        with pytest.raises(RuntimeError) as info:
            dl._lock_dir()
        assert str(info.value) == f"safe_delta_write: lock directory {expected_base!r} is owned by another user; refusing to use it"


class TestSafeWriteLocalPath:
    def test_the_filelock_package_is_required_for_local_paths(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dl, "FileLock", None)
        with pytest.raises(ImportError, match="requires the 'filelock' package; pip install filelock"):
            dl.safe_delta_write(str(tmp_path / "t"), lambda: None)

    def test_the_lock_is_acquired_and_the_result_returned(self, temp_root, fake_lock, log):
        out = dl.safe_delta_write(str(temp_root / "tbl"), lambda: "done")
        assert out == "done"
        assert fake_lock.created and fake_lock.created[0].endswith(".lock")
        assert log.debug_calls == [("Acquired lock for local Delta path: %s", str(temp_root / "tbl"))]

    def test_a_custom_lock_suffix_is_used(self, temp_root, fake_lock):
        dl.safe_delta_write(str(temp_root / "tbl"), lambda: 1, lock_suffix=".xlock")
        assert fake_lock.created[0].endswith(".xlock")

    def test_the_default_lock_timeout_is_twenty_minutes(self):
        assert inspect.signature(dl.safe_delta_write).parameters["lock_timeout"].default == 1200

    def test_a_timeout_raises_and_warns_with_the_path(self, temp_root, fake_lock, log):
        fake_lock.behaviour = "timeout"
        with pytest.raises(TimeoutError) as info:
            dl.safe_delta_write(str(temp_root / "tbl"), lambda: 1, lock_timeout=5)
        assert str(info.value) == f"safe_delta_write: could not acquire lock for {str(temp_root / 'tbl')!r} within 5s; operation was skipped"
        assert log.warning_calls == [("Timeout while waiting for lock on %s. Skipping operation.", str(temp_root / "tbl"))]

    def test_an_operation_error_is_logged_and_reraised_unchanged(self, temp_root, fake_lock, log):
        error = ValueError("boom")

        def failing():
            raise error

        with pytest.raises(ValueError) as info:
            dl.safe_delta_write(str(temp_root / "tbl"), failing)
        assert info.value is error
        assert log.exception_calls == [("Delta operation failed on %s: %s", str(temp_root / "tbl"), error)]

    def test_normalised_paths_share_one_lock_file_on_windows(self, temp_root, fake_lock):
        if os.name != "nt":
            pytest.skip("case-insensitive path normalisation applies on Windows only")
        dl.safe_delta_write(str(temp_root / "Orders"), lambda: 1)
        dl.safe_delta_write(str(temp_root / "orders"), lambda: 1)
        assert fake_lock.created[0] == fake_lock.created[1]


class TestSafeWriteNonLocalPath:
    def test_a_remote_path_runs_without_a_lock_and_warns(self, fake_lock, log):
        out = dl.safe_delta_write("s3://bucket/table", lambda: "remote")
        assert out == "remote"
        assert fake_lock.created == []
        assert log.warning_calls == [("Delta operation on non-local path: %s. Proceeding without lock.", "s3://bucket/table")]

    def test_a_remote_operation_error_is_logged_and_reraised(self, fake_lock, log):
        error = RuntimeError("remote down")

        def failing():
            raise error

        with pytest.raises(RuntimeError) as info:
            dl.safe_delta_write("s3://bucket/table", failing)
        assert info.value is error
        assert log.exception_calls == [("Delta operation failed on %s: %s", "s3://bucket/table", error)]

    def test_the_lock_timeout_is_keyword_only(self, tmp_path, fake_lock):
        with pytest.raises(TypeError):
            dl.safe_delta_write(str(tmp_path / "t"), lambda: 1, 5)

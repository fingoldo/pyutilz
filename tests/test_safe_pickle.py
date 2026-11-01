"""Sensor tests for the centralised sha256-sidecar pickle helpers.

Mirrors mlframe's tests/utils/test_safe_pickle.py -- pyutilz.core.safe_pickle is now
the canonical shared implementation mlframe re-exports from.
"""
from __future__ import annotations

import pickle

import pytest

from pyutilz.core.safe_pickle import (
    PickleVerificationError,
    safe_dump,
    safe_load,
    verify_sidecar,
    write_sidecar,
)


def _make_pickle(path: str, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def test_verify_sidecar_missing_default_fail_closed(tmp_path, monkeypatch):
    """No sidecar + no env opt-in -> verify returns False (default-strict)."""
    monkeypatch.delenv("PYUTILZ_ALLOW_UNVERIFIED_PICKLE", raising=False)
    p = tmp_path / "bundle.pkl"
    _make_pickle(str(p), {"a": 1})
    assert verify_sidecar(str(p)) is False


def test_verify_sidecar_missing_env_opt_in_passes(tmp_path, monkeypatch):
    """No sidecar + env opt-in -> verify returns True (legacy escape)."""
    monkeypatch.setenv("PYUTILZ_ALLOW_UNVERIFIED_PICKLE", "1")
    p = tmp_path / "bundle.pkl"
    _make_pickle(str(p), {"a": 1})
    assert verify_sidecar(str(p)) is True


def test_verify_sidecar_explicit_allow_unverified_overrides_env(tmp_path, monkeypatch):
    """Per-call ``allow_unverified=True`` ignores the env var (set or unset)."""
    monkeypatch.delenv("PYUTILZ_ALLOW_UNVERIFIED_PICKLE", raising=False)
    p = tmp_path / "bundle.pkl"
    _make_pickle(str(p), {"a": 1})
    assert verify_sidecar(str(p), allow_unverified=True) is True
    assert verify_sidecar(str(p), allow_unverified=False) is False


def test_verify_sidecar_matching_digest_returns_true(tmp_path):
    p = tmp_path / "bundle.pkl"
    _make_pickle(str(p), {"a": 1})
    write_sidecar(str(p))
    assert verify_sidecar(str(p)) is True


def test_verify_sidecar_mismatched_digest_returns_false(tmp_path):
    """Tampered file (digest no longer matches sidecar) returns False even
    with the env opt-in -- corrupt sidecar is a security signal, not legacy
    behaviour."""
    p = tmp_path / "bundle.pkl"
    _make_pickle(str(p), {"a": 1})
    write_sidecar(str(p))
    with open(p, "ab") as f:
        f.write(b"extra-bytes-after-pickle")
    assert verify_sidecar(str(p)) is False


def test_safe_load_round_trip_with_sidecar(tmp_path):
    p = tmp_path / "round.pkl"
    safe_dump({"k": [1, 2, 3]}, str(p))
    obj = safe_load(str(p))
    assert obj == {"k": [1, 2, 3]}


def test_safe_load_missing_sidecar_raises(tmp_path, monkeypatch):
    """Missing sidecar -> PickleVerificationError under default-strict."""
    monkeypatch.delenv("PYUTILZ_ALLOW_UNVERIFIED_PICKLE", raising=False)
    p = tmp_path / "no_sidecar.pkl"
    _make_pickle(str(p), {"x": 9})
    with pytest.raises(PickleVerificationError):
        safe_load(str(p))


def test_safe_load_missing_sidecar_env_opt_in_loads(tmp_path, monkeypatch):
    monkeypatch.setenv("PYUTILZ_ALLOW_UNVERIFIED_PICKLE", "1")
    p = tmp_path / "legacy.pkl"
    _make_pickle(str(p), {"x": 9})
    assert safe_load(str(p)) == {"x": 9}


def test_safe_load_corrupt_sidecar_raises(tmp_path, monkeypatch):
    """Digest mismatch raises even with env opt-in set."""
    monkeypatch.setenv("PYUTILZ_ALLOW_UNVERIFIED_PICKLE", "1")
    p = tmp_path / "corrupt.pkl"
    safe_dump({"k": "v"}, str(p))
    with open(p, "ab") as f:
        f.write(b"\x00")
    with pytest.raises(PickleVerificationError):
        safe_load(str(p))


def test_write_sidecar_is_idempotent(tmp_path):
    p = tmp_path / "idem.pkl"
    _make_pickle(str(p), [1, 2, 3])
    write_sidecar(str(p))
    digest_first = (tmp_path / "idem.pkl.sha256").read_text(encoding="utf-8")
    write_sidecar(str(p))
    digest_second = (tmp_path / "idem.pkl.sha256").read_text(encoding="utf-8")
    assert digest_first == digest_second


def test_custom_env_var_name_lets_embedders_keep_their_own(tmp_path, monkeypatch):
    """A project embedding this module can keep its own historical opt-in name
    (e.g. mlframe's MLFRAME_ALLOW_UNVERIFIED_PICKLE) by passing env_var=."""
    monkeypatch.delenv("PYUTILZ_ALLOW_UNVERIFIED_PICKLE", raising=False)
    monkeypatch.delenv("SOME_OTHER_PROJECT_ALLOW_UNVERIFIED", raising=False)
    p = tmp_path / "bundle.pkl"
    _make_pickle(str(p), {"a": 1})
    # PYUTILZ_ALLOW_UNVERIFIED_PICKLE is irrelevant when a custom env_var is given.
    monkeypatch.setenv("PYUTILZ_ALLOW_UNVERIFIED_PICKLE", "1")
    assert verify_sidecar(str(p), env_var="SOME_OTHER_PROJECT_ALLOW_UNVERIFIED") is False
    monkeypatch.setenv("SOME_OTHER_PROJECT_ALLOW_UNVERIFIED", "1")
    assert verify_sidecar(str(p), env_var="SOME_OTHER_PROJECT_ALLOW_UNVERIFIED") is True


import logging as _logging
import os as _os
import threading as _threading
from unittest.mock import patch as _patch

from pyutilz.core import safe_pickle as sp


def _digest_of(path):
    import hashlib

    return hashlib.sha256(open(path, "rb").read()).hexdigest()


class TestSidecarVerification:
    def test_a_missing_payload_is_refused_and_logged_by_path(self, tmp_path, caplog):
        p = tmp_path / "gone.pkl"
        (tmp_path / "gone.pkl.sha256").write_text("a" * 64 + "  gone.pkl\n")
        with caplog.at_level(_logging.ERROR, logger=sp.logger.name):
            assert verify_sidecar(str(p)) is False
        assert [r.getMessage() for r in caplog.records] == [f"verify_sidecar: payload {p} does not exist -- refusing to verify."]

    @pytest.mark.parametrize("value", ["true", "yes", "on", "1", "  TRUE  "])
    def test_each_truthy_env_value_allows_a_missing_sidecar(self, tmp_path, monkeypatch, value):
        monkeypatch.setenv("PYUTILZ_ALLOW_UNVERIFIED_PICKLE", value)
        p = tmp_path / "legacy.pkl"
        _make_pickle(str(p), {"a": 1})
        assert verify_sidecar(str(p)) is True

    @pytest.mark.parametrize("value", ["0", "no", "off", "false", ""])
    def test_a_falsy_env_value_keeps_the_sidecar_mandatory(self, tmp_path, monkeypatch, value):
        monkeypatch.setenv("PYUTILZ_ALLOW_UNVERIFIED_PICKLE", value)
        p = tmp_path / "legacy.pkl"
        _make_pickle(str(p), {"a": 1})
        assert verify_sidecar(str(p)) is False

    def test_the_allowed_warning_names_the_env_var_and_the_fix(self, tmp_path, caplog):
        p = tmp_path / "legacy.pkl"
        _make_pickle(str(p), {"a": 1})
        with caplog.at_level(_logging.WARNING, logger=sp.logger.name):
            assert verify_sidecar(str(p), allow_unverified=True) is True
        expected = (
            f"verify_sidecar: no .sha256 sidecar for {p} -- PYUTILZ_ALLOW_UNVERIFIED_PICKLE "
            "is set so load proceeds WITHOUT content verification (RCE risk if path is "
            "attacker-reachable). Generate the sidecar with `sha256sum <path> > <path>.sha256` "
            "and unset the env var to restore default fail-closed behaviour."
        )
        assert [r.getMessage() for r in caplog.records] == [expected]

    def test_the_refusal_names_the_env_var_for_legacy_bundles(self, tmp_path, caplog):
        p = tmp_path / "legacy.pkl"
        _make_pickle(str(p), {"a": 1})
        with caplog.at_level(_logging.ERROR, logger=sp.logger.name):
            assert verify_sidecar(str(p), allow_unverified=False, env_var="MY_PROJECT_VAR") is False
        expected = (
            f"verify_sidecar: no .sha256 sidecar for {p} -- refusing to load (default-strict). "
            "Generate the sidecar with `sha256sum <path> > <path>.sha256` or set "
            "MY_PROJECT_VAR=1 for legacy un-sidecar'd bundles."
        )
        assert [r.getMessage() for r in caplog.records] == [expected]

    def test_an_unreadable_sidecar_is_refused_and_logged(self, tmp_path, caplog):
        p = tmp_path / "bad.pkl"
        _make_pickle(str(p), {"a": 1})
        (tmp_path / "bad.pkl.sha256").write_bytes(b"\xff\xfe\xfd")
        with caplog.at_level(_logging.ERROR, logger=sp.logger.name):
            assert verify_sidecar(str(p)) is False
        assert any(r.getMessage().startswith(f"verify_sidecar: could not read sidecar {p}.sha256: ") for r in caplog.records)

    def test_an_empty_sidecar_is_refused_as_unreadable(self, tmp_path, caplog):
        p = tmp_path / "empty.pkl"
        _make_pickle(str(p), {"a": 1})
        (tmp_path / "empty.pkl.sha256").write_text("")
        with caplog.at_level(_logging.ERROR, logger=sp.logger.name):
            assert verify_sidecar(str(p)) is False
        assert any("could not read sidecar" in r.getMessage() for r in caplog.records)

    def test_a_sidecar_without_a_64_char_hex_digest_is_refused_and_logged(self, tmp_path, caplog):
        p = tmp_path / "short.pkl"
        _make_pickle(str(p), {"a": 1})
        sidecar = tmp_path / "short.pkl.sha256"
        sidecar.write_text("a" * 63 + "\n")
        with caplog.at_level(_logging.ERROR, logger=sp.logger.name):
            assert verify_sidecar(str(p)) is False
        assert [r.getMessage() for r in caplog.records] == [
            f"verify_sidecar: sidecar {sidecar} does not contain a 64-character hex sha256 digest -- refusing to verify."
        ]

    def test_a_non_ascii_sidecar_digest_is_refused_not_compared(self, tmp_path):
        p = tmp_path / "uni.pkl"
        _make_pickle(str(p), {"a": 1})
        (tmp_path / "uni.pkl.sha256").write_text("é" * 64 + "\n", encoding="utf-8")
        assert verify_sidecar(str(p)) is False

    def test_a_digest_one_character_too_long_is_refused(self, tmp_path):
        p = tmp_path / "long.pkl"
        _make_pickle(str(p), {"a": 1})
        (tmp_path / "long.pkl.sha256").write_text(_digest_of(str(p)) + "a\n")
        assert verify_sidecar(str(p)) is False

    def test_a_mismatched_digest_is_refused(self, tmp_path):
        p = tmp_path / "mismatch.pkl"
        _make_pickle(str(p), {"a": 1})
        (tmp_path / "mismatch.pkl.sha256").write_text("0" * 64 + "\n")
        assert verify_sidecar(str(p)) is False

    def test_the_passed_payload_bytes_are_the_ones_hashed(self, tmp_path):
        p = tmp_path / "bytes.pkl"
        _make_pickle(str(p), {"a": 1})
        write_sidecar(str(p))
        assert sp._verify_sidecar(str(p), payload=b"other bytes") is False
        assert sp._verify_sidecar(str(p), payload=open(p, "rb").read()) is True


class TestPayloadLocks:
    def test_a_lock_entry_is_dropped_once_its_last_user_leaves(self, tmp_path):
        p = str(tmp_path / "x.pkl")
        with sp._get_path_lock(p):
            pass
        assert sp._path_locks == {}

    def test_a_second_waiter_is_counted_while_the_first_holds_the_lock(self, tmp_path):
        p = str(tmp_path / "shared.pkl")
        holding = _threading.Event()
        release = _threading.Event()
        waiting = _threading.Event()

        def holder():
            with sp._get_path_lock(p):
                holding.set()
                release.wait(5)

        def waiter():
            waiting.set()
            with sp._get_path_lock(p):
                pass

        t1 = _threading.Thread(target=holder)
        t1.start()
        holding.wait(5)
        t2 = _threading.Thread(target=waiter)
        t2.start()
        waiting.wait(5)
        import time as _time

        for _ in range(200):
            entry = sp._path_locks.get(_os.path.normcase(_os.path.abspath(p)))
            if entry is not None and entry.refcount == 2:
                break
            _time.sleep(0.01)
        observed = sp._path_locks[_os.path.normcase(_os.path.abspath(p))].refcount
        release.set()
        t1.join(5)
        t2.join(5)
        assert observed == 2
        assert sp._path_locks == {}

    def test_a_lock_entry_survives_a_pickle_round_trip_with_its_count(self):
        import pickle as _pickle

        entry = sp._PathLockEntry()
        entry.refcount = 3
        restored = _pickle.loads(_pickle.dumps(entry))
        assert restored.refcount == 3
        assert restored.lock is not None


class TestTruthyAndAllowance:
    @pytest.mark.parametrize("value, expected", [("1", True), ("true", True), ("yes", True), ("on", True), ("0", False), ("no", False), ("", False)])
    def test_truthy_words_are_exactly_one_true_yes_on(self, value, expected):
        assert sp._truthy(value) is expected

    def test_a_zero_string_in_the_env_is_not_an_allowance(self, monkeypatch):
        monkeypatch.setenv("PYUTILZ_ALLOW_UNVERIFIED_PICKLE", "0")
        assert sp._allow_unverified() is False

    def test_an_unset_env_var_is_not_an_allowance(self, monkeypatch):
        monkeypatch.delenv("PYUTILZ_ALLOW_UNVERIFIED_PICKLE", raising=False)
        assert sp._allow_unverified() is False


class TestSidecarWriting:
    def test_the_sidecar_is_the_digest_then_two_spaces_then_the_name_and_a_newline(self, tmp_path):
        p = tmp_path / "named.pkl"
        _make_pickle(str(p), {"a": 1})
        write_sidecar(str(p))
        assert (tmp_path / "named.pkl.sha256").read_text() == f"{_digest_of(str(p))}  named.pkl\n"

    def test_a_failed_sidecar_replace_removes_its_temp_file(self, tmp_path):
        p = tmp_path / "t.pkl"
        _make_pickle(str(p), {"a": 1})
        with _patch("pyutilz.core.safe_pickle._replace_with_retry", side_effect=OSError("no replace")), pytest.raises(OSError):
            sp._write_sidecar_digest(str(p), "a" * 64)
        assert not [n for n in _os.listdir(tmp_path) if ".tmp." in n]

    def test_a_failed_payload_replace_removes_its_temp_file(self, tmp_path):
        p = tmp_path / "fail.pkl"
        with _patch("pyutilz.core.safe_pickle._replace_with_retry", side_effect=OSError("no replace")), pytest.raises(OSError):
            safe_dump({"a": 1}, str(p))
        assert not [n for n in _os.listdir(tmp_path) if ".tmp." in n]
        assert not p.exists()


class TestSafeLoadMessages:
    def test_a_refusal_names_the_path_the_fix_and_the_bypass(self, tmp_path):
        p = tmp_path / "nosidecar.pkl"
        _make_pickle(str(p), {"a": 1})
        with pytest.raises(PickleVerificationError) as info:
            safe_load(str(p))
        assert str(info.value) == (
            f"safe_load: refusing to unpickle {str(p)!r}; sha256 sidecar missing or mismatch. "
            "Run write_sidecar(path) on a trusted copy or set PYUTILZ_ALLOW_UNVERIFIED_PICKLE=1 to bypass (loud WARN)."
        )

    def test_a_missing_file_is_refused_as_a_verification_failure(self, tmp_path):
        with pytest.raises(PickleVerificationError):
            safe_load(str(tmp_path / "absent.pkl"))


class TestReplaceRetry:
    def test_the_retry_backs_off_by_a_growing_tenth_of_a_second(self, tmp_path):
        sleeps = []
        calls = []

        def flaky(src, dst):
            calls.append(1)
            if len(calls) < 4:
                raise PermissionError("busy")

        with _patch("pyutilz.core.safe_pickle.os.replace", side_effect=flaky), _patch("pyutilz.core.safe_pickle.time.sleep", side_effect=sleeps.append):
            sp._replace_with_retry("a", "b")
        assert sleeps == [0.01, 0.02, 0.03]
        assert len(calls) == 4

    def test_the_retry_budget_is_ten_attempts(self):
        calls = []

        def always_busy(src, dst):
            calls.append(1)
            raise PermissionError("busy")

        with _patch("pyutilz.core.safe_pickle.os.replace", side_effect=always_busy), _patch("pyutilz.core.safe_pickle.time.sleep"), pytest.raises(
            PermissionError
        ):
            sp._replace_with_retry("a", "b")
        assert len(calls) == 10

    def test_other_errors_are_not_retried(self):
        calls = []

        def boom(src, dst):
            calls.append(1)
            raise OSError("disk gone")

        with _patch("pyutilz.core.safe_pickle.os.replace", side_effect=boom), pytest.raises(OSError):
            sp._replace_with_retry("a", "b")
        assert len(calls) == 1


class TestKeywordOnlyAndSidecarShape:
    def test_verify_sidecar_refuses_a_positional_allow_unverified(self, tmp_path):
        p = tmp_path / "kw.pkl"
        _make_pickle(str(p), {"a": 1})
        with pytest.raises(TypeError):
            verify_sidecar(str(p), True)

    def test_private_verify_sidecar_refuses_a_positional_allow_unverified(self, tmp_path):
        p = tmp_path / "kw_private.pkl"
        _make_pickle(str(p), {"a": 1})
        with pytest.raises(TypeError):
            sp._verify_sidecar(str(p), True)

    def test_safe_load_refuses_a_positional_allow_unverified(self, tmp_path):
        p = tmp_path / "kw_load.pkl"
        _make_pickle(str(p), {"a": 1})
        with pytest.raises(TypeError):
            safe_load(str(p), True)

    def test_safe_dump_refuses_a_positional_protocol_and_writes_nothing(self, tmp_path):
        p = tmp_path / "kw_dump.pkl"
        with pytest.raises(TypeError):
            safe_dump({"a": 1}, str(p), 2)
        assert not p.exists()

    def test_replace_with_retry_refuses_a_positional_attempts(self):
        with _patch("pyutilz.core.safe_pickle.os.replace") as replace, pytest.raises(TypeError):
            sp._replace_with_retry("a", "b", 3)
        replace.assert_not_called()

    def test_a_sidecar_with_trailing_text_after_the_digest_is_refused_and_logged(self, tmp_path, caplog):
        p = tmp_path / "trailing.pkl"
        _make_pickle(str(p), {"a": 1})
        (tmp_path / "trailing.pkl.sha256").write_text("a" * 64 + "x  trailing.pkl\n")
        with caplog.at_level(_logging.ERROR, logger=sp.logger.name):
            assert verify_sidecar(str(p)) is False
        assert any("does not contain a 64-character hex sha256 digest" in r.getMessage() for r in caplog.records)

    @pytest.mark.skipif(_os.name == "nt", reason="POSIX file modes only")
    def test_a_safe_dump_payload_is_created_mode_0600_regardless_of_umask(self, tmp_path):
        p = tmp_path / "mode.pkl"
        old = _os.umask(0)
        try:
            safe_dump({"a": 1}, str(p))
        finally:
            _os.umask(old)
        assert _os.stat(p).st_mode & 0o777 == 0o600

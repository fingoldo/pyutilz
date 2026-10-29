"""Behaviour of ``pyutilz.system.system.fsutils.ensure_dir_exists``: a missing directory and its parents are created and the
call reports that it was created, an existing one is left as it is and reports that it already existed, and a repeated call
does not raise. Everything runs under ``tmp_path``."""

from __future__ import annotations

from pyutilz.system.system import fsutils


def test_a_missing_directory_and_its_parents_are_created(tmp_path):
    target = tmp_path / "a" / "b"
    assert fsutils.ensure_dir_exists(str(target)) is False
    assert target.is_dir()


def test_an_existing_directory_is_reported_as_existing(tmp_path):
    assert fsutils.ensure_dir_exists(str(tmp_path)) is True


def test_a_repeated_call_does_not_raise(tmp_path):
    target = str(tmp_path / "x")
    fsutils.ensure_dir_exists(target)
    assert fsutils.ensure_dir_exists(target) is True

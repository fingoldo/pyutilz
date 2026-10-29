"""Behaviour of ``pyutilz.system.system.misc``: the library version map keeps the modules that import and expose a
``__version__`` and skips the rest without raising, and a string of names is split on the given separator. The notebook
check is False in a plain process."""

from __future__ import annotations

from pyutilz.system.system import misc


def test_an_importable_library_is_mapped_to_its_version():
    import pyutilz

    assert misc.get_libs_versions("pyutilz") == {"pyutilz": pyutilz.__version__}


def test_a_missing_library_is_skipped_without_raising():
    assert "nosuch_lib_zz" not in misc.get_libs_versions(["nosuch_lib_zz"])


def test_a_string_is_split_on_the_separator():
    assert misc.get_libs_versions("nosuch_lib_zz,nosuch_lib_yy", sep=",") == {}


def test_a_plain_process_is_not_ipython():
    assert misc.run_from_ipython() is False

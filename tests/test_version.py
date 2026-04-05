import re
import pytest


def test_version_is_string():
    from pyutilz.version import __version__
    assert isinstance(__version__, str)


def test_version_matches_semver():
    from pyutilz.version import __version__
    assert re.match(r"^\d+\.\d+\.\d+$", __version__)


def test_version_importable_from_package():
    from pyutilz import __version__
    from pyutilz.version import __version__ as v2
    assert __version__ == v2

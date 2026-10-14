"""Without GPUtil, a requested GPU free-RAM minimum cannot be verified, so the check refuses it."""

import sys

from pyutilz.system.system import misc


def _no_gputil(monkeypatch):
    """Make `import GPUtil` raise ImportError, as on a host where the package is not installed."""
    monkeypatch.setitem(sys.modules, "GPUtil", None)


def test_a_requested_gpu_minimum_is_refused_when_gputil_is_missing(monkeypatch):
    _no_gputil(monkeypatch)
    assert misc._gpu_initial_conditions_ok(4.0, []) is False


def test_no_requested_gpu_minimum_passes_when_gputil_is_missing(monkeypatch):
    _no_gputil(monkeypatch)
    assert misc._gpu_initial_conditions_ok(0.0, []) is True

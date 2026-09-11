"""PT-9 — importing any pyutilz module performs no network I/O and no environment write of its own.

Catches the failure mode where a module does ``URL_BASE = requests.get('https://api.ipify.org')`` at the top level,
which blocks ``import pyutilz`` for seconds during package init, fails offline or behind a firewall, and in worse
cases sends a request to a remote service every time a downstream consumer imports the module - multiplied by N test
workers during collection.

The probe is ``py_ci_shared.import_side_effects``: a fresh interpreter per target with ``socket.socket``,
``urlopen`` and ``os.environ`` writes patched, then ``import``. A side effect a module catches and swallows still
fails - the copy this replaced printed those and exited 0, and its environment block had been written and switched
off. Only pyutilz's own code is judged: importing numpy sets and clears ``OPENBLAS_MAIN_FREE``, and urllib3 opens a
socket to test for IPv6, and neither is pyutilz's to fix.
"""

from __future__ import annotations

from py_ci_shared.import_side_effects import assert_imports_have_no_side_effects

# Sub-packages walked explicitly: the lazy proxy on ``pyutilz`` itself would otherwise hide a side effect in the
# real target until first attribute access.
_TARGETS = [
    "pyutilz",
    "pyutilz.core.pythonlib",
    "pyutilz.core.serialization",
    "pyutilz.text.strings",
    "pyutilz.text.similarity",
    "pyutilz.dev.logginglib",
    "pyutilz.dev.benchmarking",
    "pyutilz.system.parallel",
    "pyutilz.system.distributed",
    "pyutilz.system.monitoring",
    "pyutilz.data.numpylib",
    "pyutilz.llm.factory",
    "pyutilz.llm.base",
    "pyutilz.llm._retry",
]


def test_imports_have_no_side_effects_of_their_own():
    """No probed module opens a socket, calls urlopen or writes os.environ while it is imported."""
    assert_imports_have_no_side_effects(_TARGETS, first_party=("pyutilz",), block_environ=True, isolate=True, timeout=120)

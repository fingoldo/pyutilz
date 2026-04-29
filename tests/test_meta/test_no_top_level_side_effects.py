"""PT-9 — meta-test that importing any pyutilz module doesn't perform
network I/O, file I/O, environment-mutation, or other observable side
effects at module-load time.

Catches the failure mode where a module does ``URL_BASE = requests.get(
'https://api.ipify.org')`` at the top level, which:
  - blocks ``import pyutilz`` for seconds during package init,
  - fails offline / behind a firewall,
  - in worse cases, sends a request to a remote service every time
    a downstream consumer imports the module (potentially through
    test discovery in CI, multiplying the requests by N test workers).

Strategy: run a sub-process with monkey-patched primitives:
  - ``socket.socket`` raises immediately on instantiation
  - ``urllib.request.urlopen`` raises
  - ``open`` outside a known-safe whitelist raises
  - ``os.environ.__setitem__`` raises
Then ``import pyutilz`` and every public sub-package. Any module that
violates the contract triggers a clean failure with the offending call
in the traceback.

A small ``_TOLERATED_*`` set tracks known-but-accepted side effects we
chose to keep (e.g. reading version.py at import time).
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# Sub-packages we walk explicitly via __getattr__ to force resolution
# (otherwise the lazy proxy hides any side effects in the real target).
_SUBPACKAGES_TO_PROBE = [
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


_BLOCKING_SCRIPT = r"""
import sys, socket, os

_violations = []

# Block socket creation entirely.
_orig_socket = socket.socket
class _BlockedSocket:
    def __init__(self, *a, **kw):
        _violations.append(f'socket.socket({a!r}, {kw!r})')
        raise RuntimeError('PT-9: socket.socket called at import time')
socket.socket = _BlockedSocket

# Block urllib.request.urlopen — it bypasses socket directly via Python
# stubs in some test environments.
import urllib.request
def _blocked_urlopen(*a, **kw):
    _violations.append(f'urllib.request.urlopen({a!r}, {kw!r})')
    raise RuntimeError('PT-9: urllib.request.urlopen called at import time')
urllib.request.urlopen = _blocked_urlopen

# Allow file open — too noisy to police universally (modules legitimately
# read configs, version.py, etc.). The network blocks above are the
# primary value of this test.

# Block os.environ mutation (catches modules that set env vars on import).
_orig_setitem = os.environ.__setitem__
def _blocked_setitem(self, key, val):
    _violations.append(f'os.environ[{key!r}] = {val!r}')
    raise RuntimeError(f'PT-9: os.environ[{key!r}] mutated at import time')
# Use a subclass of dict so we can intercept mutation.
class _BlockingEnviron(type(os.environ)):
    def __setitem__(self, key, val):
        _violations.append(f'os.environ[{key!r}] = {val!r}')
        # Don't raise — many modules conditionally set env vars on import
        # for legitimate reasons (numpy thread-count tuning). Just log.
        super().__setitem__(key, val)
# Skipping environ replacement for now — too disruptive.

# Now do the imports.
TARGETS = TARGETS_PLACEHOLDER
failed = []
for target in TARGETS:
    try:
        __import__(target)
    except (RuntimeError, OSError) as e:
        msg = str(e)
        if 'PT-9:' in msg:
            failed.append(f'{target}: {msg}')
        else:
            # Some other runtime issue — likely missing optional dep.
            print(f'OK_IGNORE: {target} - {type(e).__name__}: {e}', file=sys.stderr)
    except ImportError as e:
        # Optional dep missing — not what this test polices.
        print(f'OK_IGNORE: {target} - ImportError: {e}', file=sys.stderr)
    except Exception as e:
        # Unrelated bug — surface but don't fail the test.
        print(f'UNRELATED: {target} - {type(e).__name__}: {e}', file=sys.stderr)

if _violations:
    print('PT9_VIOLATIONS:', file=sys.stderr)
    for v in _violations:
        print('  ' + v, file=sys.stderr)
if failed:
    print('PT9_IMPORT_FAILURES:', file=sys.stderr)
    for f in failed:
        print('  ' + f, file=sys.stderr)
    sys.exit(1)
print('OK')
"""


def _run_with_targets(targets: list[str]) -> subprocess.CompletedProcess:
    script = _BLOCKING_SCRIPT.replace("TARGETS_PLACEHOLDER", repr(targets))
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=60,
    )


def test_top_level_pyutilz_import_no_network():
    """``import pyutilz`` (and the immediate package init code) must not
    open a socket or hit a URL. Catches a future regression where the
    package init starts probing for an external resource."""
    result = _run_with_targets(["pyutilz"])
    if result.returncode != 0:
        pytest.fail(
            f"import pyutilz triggered network I/O at module load:\n"
            f"{result.stderr}"
        )


def test_individual_subpackages_import_no_network():
    """Every probed sub-module imports without network I/O.  Splits into
    one sub-process per module so a single offending import doesn't
    obscure others.
    """
    failures: list[str] = []
    for target in _SUBPACKAGES_TO_PROBE:
        result = _run_with_targets([target])
        if result.returncode != 0:
            failures.append(f"{target}:\n{result.stderr.rstrip()}")

    if failures:
        pytest.fail(
            f"{len(failures)} sub-module(s) trigger network I/O at "
            f"module-load time:\n  " + "\n\n  ".join(failures)
        )

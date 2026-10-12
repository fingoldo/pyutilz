"""Scanner tests for timing_literal_in_call: a bare duration literal at a sleep or backoff site."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit.timing_literal_in_call import scan_timing_literal_in_call

from ._helpers import _write


def test_flags_the_unnamed_retry_backoff_that_motivated_the_rule(tmp_path: Path):
    _write(
        tmp_path,
        "downloads.py",
        """
import random
from time import sleep

def fetch(attempt):
    sleep(10 * random.random())
""",
    )
    findings = scan_timing_literal_in_call(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "timing_literal_in_call"
    assert findings[0].file == "downloads.py"
    assert "10" in findings[0].detail


def test_flags_a_backoff_keyword_literal(tmp_path: Path):
    _write(
        tmp_path,
        "client.py",
        """
def make(retry):
    return Retrying(backoff=2.5, retries=3)
""",
    )
    findings = scan_timing_literal_in_call(tmp_path)
    assert len(findings) == 1


def test_allows_a_named_constant_a_parameter_default_and_index_offsets(tmp_path: Path):
    _write(
        tmp_path,
        "ok.py",
        """
from time import sleep

_POLL_SECONDS = 10

def fetch(attempt, retry_wait=10):
    sleep(_POLL_SECONDS)
    sleep(retry_wait * (attempt + 1))
    sleep(0)
""",
    )
    assert scan_timing_literal_in_call(tmp_path) == []


def test_ignores_http_timeouts_and_test_directories(tmp_path: Path):
    (tmp_path / "tests").mkdir()
    _write(tmp_path, "net.py", "def get(url):\n    return requests.get(url, timeout=15)\n")
    _write(tmp_path, "tests/test_slow.py", "import time\n\ndef test_x():\n    time.sleep(0.5)\n")
    assert scan_timing_literal_in_call(tmp_path) == []

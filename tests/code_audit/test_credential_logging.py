"""Scanner tests for credential_logging, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_credential_shaped_log_args,
)

from ._helpers import _write

# ---- credential_shaped_log_arg ---------------------------------------------


def test_credential_shaped_log_arg_unredacted_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
import logging
logger = logging.getLogger(__name__)

def f(proxy):
    logger.info(proxy)
""")
    findings = scan_credential_shaped_log_args(tmp_path)
    assert len(findings) == 1
    assert findings[0].severity == "P2"


def test_credential_shaped_log_arg_redacted_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
import logging
logger = logging.getLogger(__name__)

def f(proxy):
    redacted = proxy.split("@")[1]
    logger.info(redacted)
""")
    findings = scan_credential_shaped_log_args(tmp_path)
    assert findings == []

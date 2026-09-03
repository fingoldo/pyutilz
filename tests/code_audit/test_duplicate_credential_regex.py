"""Scanner tests for duplicate_credential_regex, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_duplicate_credential_regex,
)

from ._helpers import _write

# ---- duplicate_credential_regex -------------------------------------------


def test_duplicate_credential_regex_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
import re

_TOKEN_RE = re.compile(r"token=\\\\w+")
""")
    findings = scan_duplicate_credential_regex(tmp_path)
    assert len(findings) == 1, findings
    assert findings[0].check == "duplicate_credential_regex"
    assert findings[0].severity == "P2"


def test_duplicate_credential_regex_inside_canonical_module_is_clean(tmp_path: Path):
    _write(tmp_path, "secrets_scrub.py", """
import re

_TOKEN_RE = re.compile(r"token=\\\\w+")
""")
    findings = scan_duplicate_credential_regex(tmp_path, canonical_module_rel_paths=frozenset({"secrets_scrub.py"}))
    assert findings == []


def test_duplicate_credential_regex_non_credential_pattern_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
import re

_DATE_RE = re.compile(r"\\\\d{4}-\\\\d{2}-\\\\d{2}")
""")
    findings = scan_duplicate_credential_regex(tmp_path)
    assert findings == []

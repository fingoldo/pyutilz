"""Scanner tests for undeclared_imports, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_undeclared_imports,
)

from ._helpers import _write

# ---- undeclared_import ----------------------------------------------------


def test_undeclared_import_cross_domain_flags_p1(tmp_path: Path):
    (tmp_path / "web").mkdir()
    _write(tmp_path, "web/bad.py", """
import pandas as pd

def f():
    return pd.DataFrame()
""")
    findings = scan_undeclared_imports(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "undeclared_import"
    assert findings[0].severity == "P1"


def test_undeclared_import_own_domain_is_clean(tmp_path: Path):
    (tmp_path / "web").mkdir()
    _write(tmp_path, "web/ok.py", """
import requests

def f():
    return requests.get("http://x", timeout=5)
""")
    findings = scan_undeclared_imports(tmp_path)
    assert findings == []


# ---- F69: domains are matched on path segments --------------------------------------


def test_undeclared_imports_domain_matching_respects_path_boundaries():
    from pyutilz.dev.code_audit.undeclared_imports import _domain_for

    assert _domain_for("web/x.py") == "web"
    assert _domain_for("webhooks.py") is None
    assert _domain_for("dev/y.py") == "dev"
    assert _domain_for("developer_notes.py") is None

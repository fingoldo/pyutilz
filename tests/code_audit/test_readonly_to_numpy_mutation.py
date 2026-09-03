"""Scanner tests for readonly_to_numpy_mutation, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_readonly_to_numpy_mutation,
)

from ._helpers import _write

# ---- readonly_to_numpy_mutation -------------------------------------------


def test_readonly_to_numpy_mutation_fill_diagonal_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
import numpy as np

def dataset_diagnostics(df):
    C = df.corr().to_numpy()
    np.fill_diagonal(C, 0.0)
    return C
""")
    findings = scan_readonly_to_numpy_mutation(tmp_path)
    assert len(findings) == 1, findings
    f = findings[0]
    assert f.check == "readonly_to_numpy_mutation"
    assert f.severity == "P2"
    assert "C" in f.detail


def test_readonly_to_numpy_mutation_copy_true_not_flagged(tmp_path: Path):
    _write(tmp_path, "ok.py", """
import numpy as np

def dataset_diagnostics(df):
    C = df.corr().to_numpy(copy=True)
    np.fill_diagonal(C, 0.0)
    return C
""")
    findings = scan_readonly_to_numpy_mutation(tmp_path)
    assert findings == []


def test_readonly_to_numpy_mutation_copyto_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
import numpy as np

def f(df, other):
    C = df.to_numpy()
    np.copyto(C, other)
""")
    findings = scan_readonly_to_numpy_mutation(tmp_path)
    assert len(findings) == 1, findings


def test_readonly_to_numpy_mutation_fresh_numpy_array_not_flagged(tmp_path: Path):
    """A ``fill_diagonal`` target that never came from ``.to_numpy()`` (e.g. built via
    ``np.corrcoef``) is out of scope for this scanner."""
    _write(tmp_path, "ok.py", """
import numpy as np

def f(X):
    C = np.corrcoef(X, rowvar=False)
    np.fill_diagonal(C, 0.0)
    return C
""")
    findings = scan_readonly_to_numpy_mutation(tmp_path)
    assert findings == []


def test_readonly_to_numpy_mutation_unrelated_to_numpy_not_flagged(tmp_path: Path):
    """An uncopied ``.to_numpy()`` result that is never passed to an in-place mutator is fine."""
    _write(tmp_path, "ok.py", """
def f(df):
    arr = df.to_numpy()
    return arr.sum()
""")
    findings = scan_readonly_to_numpy_mutation(tmp_path)
    assert findings == []


# ---- F96: readonly to_numpy mutation, scoped per function ---------------------------


def test_readonly_to_numpy_mutation_reports_a_nested_def_once(tmp_path: Path):
    _write(tmp_path, "a.py", """
import numpy as np


def outer(df):
    def inner():
        A = df.to_numpy()
        np.fill_diagonal(A, 0.0)

    inner()
""")
    assert len(scan_readonly_to_numpy_mutation(tmp_path)) == 1


def test_readonly_to_numpy_mutation_does_not_leak_a_name_out_of_a_nested_scope(tmp_path: Path):
    _write(tmp_path, "a.py", """
import numpy as np


def outer(df):
    def helper():
        C = df.to_numpy()
        return C

    C = 1
    np.fill_diagonal(C, 0.0)
""")
    assert scan_readonly_to_numpy_mutation(tmp_path) == []

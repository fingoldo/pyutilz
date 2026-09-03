"""Scanner tests for test_asserts_against_production_constant, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit.test_asserts_against_production_constant import scan_test_asserts_against_production_constant

from ._helpers import _write

# ---- test_asserts_against_production_constant ----------------------------


def test_asserts_against_production_constant_flags_a_rederived_expectation(tmp_path: Path):
    """The expected value is computed from the constant the code under test reads.

    Found in the scraper codebase in its purest form: `expected = min_days * MULT` followed by
    `assert expected == min_days * MULT` -- a variable compared with its own definition, which
    never called production code at all.
    """
    _write(
        tmp_path,
        "test_backoff.py",
        """
from throttling import BASE_DELAY, compute_backoff


def test_backoff():
    assert compute_backoff(3) == BASE_DELAY * 2 ** 3
""",
    )
    findings = scan_test_asserts_against_production_constant(tmp_path)
    assert len(findings) == 1, findings
    assert "BASE_DELAY" in findings[0].detail


def test_asserts_against_production_constant_accepts_a_literal(tmp_path: Path):
    """The form this rule asks for: change the constant and the test has to change too."""
    _write(
        tmp_path,
        "test_backoff.py",
        """
from throttling import BASE_DELAY, compute_backoff


def test_backoff():
    assert compute_backoff(3) == 24
    assert BASE_DELAY == 3
""",
    )
    assert scan_test_asserts_against_production_constant(tmp_path) == []


def test_asserts_against_production_constant_accepts_a_boundary_argument(tmp_path: Path):
    """A constant feeding the code under test is not standing in for its answer.

    `choose_backend(MIN_ROWS - 1, ...) == "numpy"` is a correct boundary test whose expected value
    is the literal on the other side, and that shape was six of this rule's first sixteen hits.
    """
    _write(
        tmp_path,
        "test_dispatch.py",
        """
from kernels import MIN_ROWS, choose_backend


def test_below_the_floor():
    assert choose_backend(MIN_ROWS - 1) == "numpy"
""",
    )
    assert scan_test_asserts_against_production_constant(tmp_path) == []


def test_asserts_against_production_constant_accepts_a_path_join(tmp_path: Path):
    """`CHECKPOINT_DIR / "x.jsonl"` is path construction wearing an operator, not arithmetic."""
    _write(
        tmp_path,
        "test_checkpoint.py",
        """
from rescan import CHECKPOINT_DIR, checkpoint_path


def test_path():
    assert checkpoint_path("01abc") == CHECKPOINT_DIR / "~01abc.jsonl"
""",
    )
    assert scan_test_asserts_against_production_constant(tmp_path) == []


def test_asserts_against_production_constant_ignores_a_local_constant(tmp_path: Path):
    """A constant the test defines for itself is the expected value, spelled once."""
    _write(
        tmp_path,
        "test_backoff.py",
        """
from throttling import compute_backoff

BASE = 3


def test_backoff():
    assert compute_backoff(3) == BASE * 2 ** 3
""",
    )
    assert scan_test_asserts_against_production_constant(tmp_path) == []


def test_asserts_against_production_constant_ignores_two_derivations(tmp_path: Path):
    """Both sides derived from the same constant is a different (and rarer) mistake.

    Reporting it here would blur the message, which is about an expected value that moves with the
    implementation -- here neither side is the thing under test.
    """
    _write(
        tmp_path,
        "test_backoff.py",
        """
from throttling import BASE_DELAY


def test_algebra():
    assert BASE_DELAY * 4 == BASE_DELAY * 2 * 2
""",
    )
    assert scan_test_asserts_against_production_constant(tmp_path) == []


def test_asserts_against_production_constant_only_reads_test_files(tmp_path: Path):
    """Production code deriving one constant from another is ordinary, not a broken assertion."""
    _write(
        tmp_path,
        "throttling.py",
        """
from config import BASE_DELAY

MAX_DELAY = BASE_DELAY * 60


def check():
    assert MAX_DELAY == BASE_DELAY * 60
""",
    )
    assert scan_test_asserts_against_production_constant(tmp_path) == []


# ---- F87/F88/F177/F178/F179: asserts against a production constant ------------------


def test_asserts_against_production_constant_accepts_an_fstring_path_segment(tmp_path: Path):
    _write(tmp_path, "test_a.py", "from prod import CHECKPOINT_DIR\n\n\ndef test_p(name, p):\n    assert p == CHECKPOINT_DIR / f'{name}.jsonl'\n")
    assert scan_test_asserts_against_production_constant(tmp_path) == []


def test_asserts_against_production_constant_accepts_a_variable_path_segment(tmp_path: Path):
    _write(tmp_path, "test_a.py", """
from prod import CHECKPOINT_DIR


def test_p(sub, p):
    assert p == CHECKPOINT_DIR / sub
""")
    assert scan_test_asserts_against_production_constant(tmp_path) == []


def test_asserts_against_production_constant_looks_through_pytest_approx(tmp_path: Path):
    _write(tmp_path, "test_a.py", """
import pytest

from prod import BASE_DELAY


def test_b():
    assert backoff(3) == pytest.approx(BASE_DELAY * 2)
""")
    assert len(scan_test_asserts_against_production_constant(tmp_path)) == 1


def test_asserts_against_production_constant_ignores_a_non_test_helper_receiver(tmp_path: Path):
    _write(tmp_path, "test_a.py", """
from prod import BASE_DELAY


def helper(recorder, x):
    recorder.assertEqual(BASE_DELAY * 2, x)
""")
    assert scan_test_asserts_against_production_constant(tmp_path) == []


def test_asserts_against_production_constant_sees_a_negated_rederivation(tmp_path: Path):
    _write(tmp_path, "test_a.py", """
from prod import BASE_DELAY


def test_b():
    assert backoff(3) == -BASE_DELAY
""")
    assert len(scan_test_asserts_against_production_constant(tmp_path)) == 1


def test_asserts_against_production_constant_reports_the_offending_argument_line(tmp_path: Path):
    _write(tmp_path, "test_a.py", """
import unittest

from prod import BASE_DELAY


class T(unittest.TestCase):
    def test_b(self):
        self.assertEqual(
            backoff(3),
            BASE_DELAY * 2,
        )
""")
    findings = scan_test_asserts_against_production_constant(tmp_path)
    assert len(findings) == 1 and findings[0].line == 10

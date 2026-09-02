"""Tests for the additive-epsilon-denominator checker, distilled from the 2026-09-01 wide audit.

The detector is exercised against an injected pad of every shape it claims to catch AND against ordinary
arithmetic it must not flag -- `x / (n + 1)` is Laplace smoothing, not a degeneracy pad. Tests use tmp_path
so each scan runs against a hermetic tree.
"""

from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import scan_additive_epsilon_denominator


def _write(tmp_path: Path, name: str, source: str) -> Path:
    """Write ``source`` to ``tmp_path/name`` and return the path."""
    p = tmp_path / name
    p.write_text(source, encoding="utf-8")
    return p


# --- additive-epsilon denominators ---------------------------------------


def test_additive_epsilon_inline_pad_flagged(tmp_path: Path):
    """A `+ 1e-12` written directly in the denominator is flagged."""
    _write(tmp_path, "bad.py", "def f(a, d):\n    return a / (d + 1e-12)\n")
    findings = scan_additive_epsilon_denominator(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "additive_epsilon_denominator"
    assert findings[0].severity == "P1"


def test_additive_epsilon_reversed_operand_order_flagged(tmp_path: Path):
    """The pad is caught on either side of the `+`."""
    _write(tmp_path, "bad.py", "def f(a, d):\n    return a / (1e-9 + d)\n")
    assert len(scan_additive_epsilon_denominator(tmp_path)) == 1


def test_additive_epsilon_via_intermediate_name_flagged(tmp_path: Path):
    """A denominator bound to a padded sum one statement earlier is still the same defect."""
    _write(tmp_path, "bad.py", "def f(a, var):\n    denom = var + 1e-12\n    return a / denom\n")
    assert len(scan_additive_epsilon_denominator(tmp_path)) == 1


def test_additive_epsilon_ordinary_smoothing_is_clean(tmp_path: Path):
    """`x / (n + 1)` is Laplace smoothing and `+ 0.5` is a real term -- neither is a degeneracy pad."""
    _write(tmp_path, "ok.py", "def f(x, n, d):\n    return x / (n + 1) + x / (d + 0.5)\n")
    assert scan_additive_epsilon_denominator(tmp_path) == []


def test_additive_epsilon_explicit_degeneracy_branch_is_clean(tmp_path: Path):
    """The sanctioned form -- an explicit threshold branch -- is not flagged."""
    _write(tmp_path, "ok.py", "def f(x, d):\n    return x / d if d > 1e-9 else 0.0\n")
    assert scan_additive_epsilon_denominator(tmp_path) == []


def test_additive_epsilon_threshold_is_configurable(tmp_path: Path):
    """Raising `epsilon_max` widens what counts as a pad."""
    _write(tmp_path, "maybe.py", "def f(x, d):\n    return x / (d + 0.01)\n")
    assert scan_additive_epsilon_denominator(tmp_path) == []
    assert len(scan_additive_epsilon_denominator(tmp_path, epsilon_max=0.1)) == 1

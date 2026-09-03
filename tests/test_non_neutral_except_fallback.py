"""Tests for the non-neutral-except-fallback checker, distilled from the 2026-09-01 wide audit.

Covers each substitution shape that disables the check it feeds, and the three exemptions that keep the
checker usable: an audible handler, an ImportError-only handler, and a `None` that reads as 'no answer'
rather than as a measurement. Tests use tmp_path so each scan runs against a hermetic tree.
"""

from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import scan_non_neutral_except_fallback


def _write(tmp_path: Path, name: str, source: str) -> Path:
    """Write ``source`` to ``tmp_path/name`` and return the path."""
    p = tmp_path / name
    p.write_text(source, encoding="utf-8")
    return p


# --- non-neutral except fallbacks ----------------------------------------


def test_non_neutral_except_silent_zero_flagged(tmp_path: Path):
    """A debug-logged handler substituting 0.0 is flagged: 0.0 is the BEST possible max error."""
    src = (
        "import logging\n"
        "logger = logging.getLogger(__name__)\n\n\n"
        "def f(a, b):\n"
        "    try:\n"
        "        return compute(a, b)\n"
        "    except Exception as e:\n"
        "        logger.debug('failed: %s', e)\n"
        "        return 0.0\n"
    )
    _write(tmp_path, "bad.py", src)
    findings = scan_non_neutral_except_fallback(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "non_neutral_except_fallback"


def test_non_neutral_except_true_from_guard_flagged(tmp_path: Path):
    """`return True` from a failed permission probe is the value that ALLOWS the action."""
    src = "def can_upload():\n    try:\n        return probe()\n    except Exception:\n        return True\n"
    _write(tmp_path, "bad.py", src)
    assert len(scan_non_neutral_except_fallback(tmp_path)) == 1


def test_non_neutral_except_negative_inf_assignment_flagged(tmp_path: Path):
    """`-np.inf` assigned in a handler guarantees the comparison's outcome."""
    src = "import numpy as np\n\n\ndef f():\n    try:\n        score = solve()\n    except Exception:\n        score = -np.inf\n    return score\n"
    _write(tmp_path, "bad.py", src)
    assert len(scan_non_neutral_except_fallback(tmp_path)) == 1


def test_non_neutral_except_audible_warning_is_clean(tmp_path: Path):
    """A handler that warns has made the failure visible; the substitution is then a documented choice."""
    src = (
        "import logging\n"
        "logger = logging.getLogger(__name__)\n\n\n"
        "def f():\n"
        "    try:\n"
        "        return compute()\n"
        "    except Exception as e:\n"
        "        logger.warning('failed: %s', e)\n"
        "        return 0.0\n"
    )
    _write(tmp_path, "ok.py", src)
    assert scan_non_neutral_except_fallback(tmp_path) == []


def test_non_neutral_except_reraise_is_clean(tmp_path: Path):
    """Re-raising is the loudest possible response, not a silent substitution."""
    src = "def f():\n    try:\n        return compute()\n    except Exception:\n        raise\n"
    _write(tmp_path, "ok.py", src)
    assert scan_non_neutral_except_fallback(tmp_path) == []


def test_non_neutral_except_none_substitution_is_clean(tmp_path: Path):
    """`None` reads as 'no answer' rather than as a measurement, so it is exempt."""
    src = "def f():\n    try:\n        return compute()\n    except Exception:\n        return None\n"
    _write(tmp_path, "ok.py", src)
    assert scan_non_neutral_except_fallback(tmp_path) == []


def test_non_neutral_except_import_error_only_is_clean(tmp_path: Path):
    """An absent optional dependency is permanent and expected; its substitution is the intended answer."""
    src = "def has_gpu():\n    try:\n        import cupy\n    except ImportError:\n        return False\n    return True\n"
    _write(tmp_path, "ok.py", src)
    assert scan_non_neutral_except_fallback(tmp_path) == []


def test_non_neutral_except_throttle_helper_counts_as_audible(tmp_path: Path):
    """A project's throttled-logging helper is audible once named in `audible_functions`."""
    src = (
        "def f():\n"
        "    try:\n"
        "        return compute()\n"
        "    except Exception as e:\n"
        "        log_throttle(logger, 'k', 30, 'failed: %s', e)\n"
        "        return 0.0\n"
    )
    _write(tmp_path, "maybe.py", src)
    assert scan_non_neutral_except_fallback(tmp_path) == []
    assert len(scan_non_neutral_except_fallback(tmp_path, audible_functions=frozenset())) == 1

def test_a_print_counts_as_saying_so(tmp_path):
    """A line on stdout distinguishes the substituted value from a real result.

    For a CLI script that is the notification channel, not a lapse. Three handlers in a harvest
    script that printed exactly what they substituted and why were reported as silent.
    """
    (tmp_path / "harvest.py").write_text(
        """
def parts_done(state):
    value = state.get("parts_done_count", 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        print(f"parts_done_count corrupted ({value!r}); resetting to 0")
        return 0
""",
        encoding="utf-8",
    )
    assert scan_non_neutral_except_fallback(tmp_path) == []


def test_a_silent_substitution_is_still_reported(tmp_path):
    """The same handler without the print is the finding, so the exemption is not a hole."""
    (tmp_path / "harvest.py").write_text(
        """
def parts_done(state):
    value = state.get("parts_done_count", 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
""",
        encoding="utf-8",
    )
    assert len(scan_non_neutral_except_fallback(tmp_path)) == 1

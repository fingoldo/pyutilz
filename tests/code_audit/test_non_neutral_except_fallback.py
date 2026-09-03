"""Scanner tests for non_neutral_except_fallback, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit.non_neutral_except_fallback import scan_non_neutral_except_fallback

from ._helpers import _write

# ---- F19/F75/F167: non-neutral except fallback --------------------------------------


def test_non_neutral_except_fallback_ignores_a_nested_callback(tmp_path: Path):
    _write(tmp_path, "m.py", """
def f():
    try:
        g()
    except ValueError:
        def cb():
            return 0.0
        register(cb)
        raise
""")
    assert scan_non_neutral_except_fallback(tmp_path) == []


def test_non_neutral_except_fallback_sees_annotated_and_tuple_fallbacks(tmp_path: Path):
    _write(tmp_path, "m.py", """
def a():
    try:
        g()
    except Exception:
        v: float = 0.0
    return v


def b():
    try:
        g()
    except Exception:
        p, q = 0.0, 0.0
    return p
""")
    assert len(scan_non_neutral_except_fallback(tmp_path)) == 2


def test_non_neutral_except_fallback_names_the_first_substitution(tmp_path: Path):
    _write(tmp_path, "m.py", """
def f(k):
    try:
        g()
    except Exception:
        if k:
            return 1.0
        return 2.0
""")
    findings = scan_non_neutral_except_fallback(tmp_path)
    assert len(findings) == 1 and "returns 1.0" in findings[0].detail


# ---- 2026-09-03 downstream-scan narrowing: measured false-positive shapes -------------
#
# Each negative below is a shape sampled from real code (pyutilz 41 findings read in full,
# 30 of mlframe/src's 446 sampled, py-ci-shared's 2 read in full) and sits beside a positive
# that must keep firing, so a later widening of the rule cannot quietly re-admit the noise.


def test_non_neutral_except_fallback_exempts_an_optional_dependency_probe(tmp_path: Path):
    """NEGATIVE. `try: import numba / _HAS = True / except Exception: _HAS = False` is the canonical
    optional-dependency probe: an absent-or-unusable dependency is a permanent, expected condition
    and `False` IS the answer. The probes catch broadly on purpose (a broken-but-installed numba
    raises something other than ImportError), which is why the pre-existing ImportError-only
    exemption missed them. 9 of pyutilz's 41 and 12 of the 30 sampled mlframe sites."""
    _write(tmp_path, "m.py", """
try:
    import numba as _numba
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False


def torch_available():
    try:
        import torch
        return True
    except Exception:
        return False
""")
    assert scan_non_neutral_except_fallback(tmp_path) == []


def test_non_neutral_except_fallback_still_fires_on_a_fail_open_gate_next_to_a_probe(tmp_path: Path):
    """POSITIVE beside the negative above. py-ci-shared's real defect: `_loc` returns 0 LOC on
    OSError, so any file the walker cannot open is scored as 0 and exempted from the size gate.
    No import in the guarded body, and OSError is an environment failure, not a determinate fact."""
    _write(tmp_path, "m.py", """
try:
    import numba
    _HAS = True
except Exception:
    _HAS = False


def _loc(path):
    try:
        with path.open("r") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0
""")
    findings = scan_non_neutral_except_fallback(tmp_path)
    assert len(findings) == 1 and "returns 0" in findings[0].detail


def test_non_neutral_except_fallback_exempts_a_total_function_parse_guard(tmp_path: Path):
    """NEGATIVE. A short straight-line body guarded by an exception that states a determinate fact
    about the INPUT (this string is not a number, this key/attribute is not there, this file does
    not exist) is the negative branch of a total function; the fallback IS the answer. Measured on
    `_safe_float`, `is_float`, `_async_sweep_start_delay`, `run_from_ipython`, and the
    `except FileNotFoundError` idempotent-delete pair in mlframe's cache stores."""
    _write(tmp_path, "m.py", """
def _safe_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _delay():
    try:
        return max(0.0, float(os.environ.get("D", "10.0")))
    except ValueError:
        return 10.0


def invalidate(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        return False
""")
    assert scan_non_neutral_except_fallback(tmp_path) == []


def test_non_neutral_except_fallback_fires_when_the_guard_is_not_total(tmp_path: Path):
    """POSITIVE beside the negative above -- the three ways a parse-guard exemption must NOT apply.
    (a) the handler also catches an environment failure (`OSError`), (b) the guarded body carries
    control flow, so the handler can fire from several places and the value means several things,
    (c) the exception is broad. All three are shapes that produced real findings in the sample."""
    _write(tmp_path, "m.py", """
def a(p):
    try:
        return int(read(p))
    except (ValueError, OSError):
        return 0


def b(rows):
    try:
        for r in rows:
            if r:
                total = float(r)
        return total
    except ValueError:
        return 0.0


def c(design, yv):
    try:
        beta = solve(design, yv)
    except Exception:
        return 0.0
    return beta
""")
    assert len(scan_non_neutral_except_fallback(tmp_path)) == 3

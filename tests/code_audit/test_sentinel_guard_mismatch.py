"""Scanner tests for sentinel_guard_mismatch, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit.sentinel_guard_mismatch import scan_sentinel_guard_mismatch

from ._helpers import _write

# ---- sentinel_guard_mismatch ---------------------------------------------
#
# A failure path returns a falsy value while the caller guards `is None`, so the failure reads as
# a legitimate answer. One transient HTML error page retired a discovery source permanently.


def test_sentinel_guard_mismatch_flags_the_canonical_case(tmp_path: Path):
    _write(
        tmp_path,
        "cdx.py",
        """
def fetch_num_pages(url):
    try:
        return int(get(url).text)
    except ValueError:
        return 0
""",
    )
    _write(
        tmp_path,
        "driver.py",
        """
def run(state):
    pages_total = fetch_num_pages(state.url)
    if pages_total is None or (pages_total > 0 and state.done):
        return
    mark_source_empty(state)
""",
    )
    findings = scan_sentinel_guard_mismatch(tmp_path)
    assert len(findings) == 1
    assert "fetch_num_pages" in findings[0].detail


def test_sentinel_guard_mismatch_allows_none_as_a_third_answer(tmp_path: Path):
    """The accepted fix for this shape was, verbatim, to make None a third answer. A function that
    returns None for the failure and a falsy value for a real outcome is the FIXED form."""
    _write(
        tmp_path,
        "cdx.py",
        """
def fetch_num_pages(url):
    try:
        return int(get(url).text)
    except ValueError:
        return None
""",
    )
    _write(
        tmp_path,
        "driver.py",
        """
def run(state):
    pages_total = fetch_num_pages(state.url)
    if pages_total is None:
        return
""",
    )
    assert scan_sentinel_guard_mismatch(tmp_path) == []


def test_sentinel_guard_mismatch_ignores_a_falsy_return_on_the_ordinary_path(tmp_path: Path):
    """A function returning 0 from its normal path is returning a number, not signalling."""
    _write(
        tmp_path,
        "counts.py",
        """
def how_many(items):
    if not items:
        return 0
    return len(items)
""",
    )
    _write(
        tmp_path,
        "driver.py",
        """
def run(items):
    n = how_many(items)
    if n is None:
        return
""",
    )
    assert scan_sentinel_guard_mismatch(tmp_path) == []


def test_sentinel_guard_mismatch_needs_a_caller_that_guards_on_none(tmp_path: Path):
    """Returning 0 on failure is fine if nobody tests the result for None."""
    _write(
        tmp_path,
        "cdx.py",
        """
def fetch_num_pages(url):
    try:
        return int(get(url).text)
    except ValueError:
        return 0
""",
    )
    _write(
        tmp_path,
        "driver.py",
        """
def run(state):
    pages_total = fetch_num_pages(state.url)
    if pages_total > 0:
        go(pages_total)
""",
    )
    assert scan_sentinel_guard_mismatch(tmp_path) == []


# ---- F21: the minus-one sentinel ----------------------------------------------------


def test_sentinel_guard_mismatch_sees_a_negative_one_sentinel(tmp_path: Path):
    _write(tmp_path, "a.py", """
def get_count(c):
    try:
        return c.fetch()
    except Exception:
        return -1


def use(c):
    n = get_count(c)
    if n is None:
        return 0
    return n
""")
    assert len(scan_sentinel_guard_mismatch(tmp_path)) == 1


def test_sentinel_guard_mismatch_accepts_a_matching_guard(tmp_path: Path):
    _write(tmp_path, "a.py", """
def get_count(c):
    try:
        return c.fetch()
    except Exception:
        return -1


def use(c):
    n = get_count(c)
    if n < 0:
        return 0
    return n
""")
    assert scan_sentinel_guard_mismatch(tmp_path) == []

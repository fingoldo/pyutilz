"""Scanner tests for accumulator_helper_bypassed, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit.accumulator_helper_bypassed import scan_accumulator_helper_bypassed

from ._helpers import HELPER, _write

# ---- accumulator_helper_bypassed -----------------------------------------


def test_accumulator_helper_bypassed_flags_a_sibling_module(tmp_path: Path):
    """The canonical shape: `_inc_stat` in a mixin, a direct `+=` in its sibling.

    Package-wide on purpose -- a per-file rule saw a helper with no bypasses and bypasses with no
    helper, and reported nothing on the very defect it was written for.
    """
    _write(tmp_path, "stats_mixin.py", HELPER)
    _write(
        tmp_path,
        "parallel_mixin.py",
        """
class Parallel:
    def paginate(self, ids):
        self.stats["total_paginated"] += len(ids)
""",
    )
    findings = scan_accumulator_helper_bypassed(tmp_path)
    assert len(findings) == 1, findings
    assert "_inc_stat" in findings[0].detail


def test_accumulator_helper_bypassed_accepts_a_write_under_the_lock(tmp_path: Path):
    """The helper here is a lock plus the write, so a caller already holding it skips nothing."""
    _write(tmp_path, "stats_mixin.py", HELPER)
    _write(
        tmp_path,
        "parallel_mixin.py",
        """
class Parallel:
    def paginate(self, ids):
        with self._lock:
            self.stats["total_paginated"] += len(ids)
""",
    )
    assert scan_accumulator_helper_bypassed(tmp_path) == []


def test_accumulator_helper_bypassed_accepts_assigning_a_measurement(tmp_path: Path):
    """`stats["root_total_count"] = count` stores a value just computed; routing it through an
    incrementing helper would be wrong, not safer. All four surviving hits in one codebase were
    this shape."""
    _write(tmp_path, "stats_mixin.py", HELPER)
    _write(
        tmp_path,
        "crawler.py",
        """
class Crawler:
    def crawl(self, count):
        self.stats["root_total_count"] = count
""",
    )
    assert scan_accumulator_helper_bypassed(tmp_path) == []


def test_accumulator_helper_bypassed_accepts_a_test_fixture(tmp_path: Path):
    """A test arranging state before asserting on it is building a fixture, not bypassing."""
    _write(tmp_path, "stats_mixin.py", HELPER)
    _write(
        tmp_path,
        "test_crawler.py",
        """
def test_pagination(c):
    c.stats["total_paginated"] += 1
""",
    )
    assert scan_accumulator_helper_bypassed(tmp_path) == []


def test_accumulator_helper_bypassed_ignores_a_local_accumulator(tmp_path: Path):
    """A local is shared with nobody, so writing it directly bypasses nothing.

    Two earlier versions of this test proved nothing and were replaced: one used
    `findings.append(...)`, which stopped being a candidate at all once the rule narrowed to
    accumulation, and one defined a helper no other function called, which the rule skips before
    it ever reaches the shared-structure question.
    """
    _write(
        tmp_path,
        "scanner.py",
        """
def collect(key):
    counts = {}
    counts[key] += 1
    return counts


def other():
    counts = {}
    counts["fixed"] += 1
    return counts


def run(key):
    return collect(key), other()
""",
    )
    assert scan_accumulator_helper_bypassed(tmp_path) == []


def test_accumulator_helper_bypassed_needs_a_parameter_keyed_owner(tmp_path: Path):
    """With every writer using a literal key there is no helper, and nobody bypasses anything.

    `record` here mutates two structures, so it can never be an owner itself -- which is what
    makes it the site the rule WOULD report if it accepted a literal-keyed writer as the owner.
    """
    _write(
        tmp_path,
        "crawler.py",
        """
class Crawler:
    def paginate(self, ids):
        self.stats["total_paginated"] += len(ids)

    def record(self, ids, seen):
        self.stats["total_dup"] += len(ids)
        self.audit_log["last"] = seen

    def run(self, ids, seen):
        self.paginate(ids)
        self.record(ids, seen)
""",
    )
    assert scan_accumulator_helper_bypassed(tmp_path) == []

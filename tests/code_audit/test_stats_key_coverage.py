"""Scanner tests for stats_key_coverage, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit.stats_key_coverage import scan_stats_key_coverage

from ._helpers import _write

# ---- stats_key_coverage --------------------------------------------------
#
# The audited crawler recorded two incidents on one dict: a lazily-created counter that was
# cumulative since process start while every sibling was per-cycle, and an unregistered key that
# turned an increment helper into a KeyError. A third happened while this rule was being written.


def test_stats_key_coverage_flags_an_undeclared_accumulating_key(tmp_path: Path):
    _write(
        tmp_path,
        "crawler.py",
        """
class Crawler:
    def _reset_stats(self):
        self.stats = {"pages": 0, "dups": 0}

    def note(self, n):
        self.stats["skipped_small"] = self.stats.get("skipped_small", 0) + n
""",
    )
    findings = scan_stats_key_coverage(tmp_path)
    assert len(findings) == 1
    assert "skipped_small" in findings[0].detail


def test_stats_key_coverage_matches_across_mixins(tmp_path: Path):
    """The real shape: one class declares the dict, another increments it, and they are one
    object only at runtime. Scoped per class, this rule missed the bug it was written from."""
    _write(
        tmp_path,
        "stats_mixin.py",
        """
class StatsMixin:
    def _reset_stats(self):
        self.stats = {"pages": 0}
""",
    )
    _write(
        tmp_path,
        "split_mixin.py",
        """
class SplitMixin:
    def split(self):
        self._inc_stat("overlapping_axis_skipped")
""",
    )
    findings = scan_stats_key_coverage(tmp_path)
    assert len(findings) == 1
    assert "overlapping_axis_skipped" in findings[0].detail


def test_stats_key_coverage_ignores_a_plain_assignment(tmp_path: Path):
    """`self.stats["k"] = value` overwrites completely every cycle, so it is safe undeclared.
    Four such keys in the audited crawler were this rule's only false positives."""
    _write(
        tmp_path,
        "crawler.py",
        """
class Crawler:
    def _reset_stats(self):
        self.stats = {"pages": 0}

    def finish(self, recovered):
        self.stats["recovered"] = recovered
""",
    )
    assert scan_stats_key_coverage(tmp_path) == []


def test_stats_key_coverage_ignores_a_declared_key(tmp_path: Path):
    _write(
        tmp_path,
        "crawler.py",
        """
class Crawler:
    def _reset_stats(self):
        self.stats = {"pages": 0, "skipped_small": 0}

    def note(self, n):
        self.stats["skipped_small"] += n
""",
    )
    assert scan_stats_key_coverage(tmp_path) == []


def test_stats_key_coverage_ignores_a_class_that_declares_nothing(tmp_path: Path):
    """A dict with no declared shape has no contract to violate."""
    _write(
        tmp_path,
        "loose.py",
        """
class Loose:
    def note(self, n):
        self.counts["whatever"] = self.counts.get("whatever", 0) + n
""",
    )
    assert scan_stats_key_coverage(tmp_path) == []


# ---- F23/F84: stats key coverage ----------------------------------------------------


def test_stats_key_coverage_reads_an_annotated_reset(tmp_path: Path):
    _write(tmp_path, "a.py", """
class C:
    def _reset(self):
        self.stats: dict = {"a": 0}

    def bump(self):
        self.stats["zz"] += 1
""")
    assert len(scan_stats_key_coverage(tmp_path)) == 1


def test_stats_key_coverage_does_not_count_a_bare_get_as_a_write(tmp_path: Path):
    _write(tmp_path, "a.py", """
class C:
    def _reset(self):
        self.stats = {"a": 0}

    def peek(self):
        return self.stats.get("never_written_key", 0)
""")
    assert scan_stats_key_coverage(tmp_path) == []

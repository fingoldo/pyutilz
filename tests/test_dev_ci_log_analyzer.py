"""Unit tests for the CI log analyzer's parsing/aggregation half.

Everything here exercises ``analyze_log_text`` and ``CIReport.render`` against
literal log text -- no ``gh`` invocation, no network. The network-facing
helpers (``_gh_json`` / ``_fetch_job_log`` / ``analyze_run``) are thin
``subprocess.run`` wrappers around the ``gh`` CLI and are deliberately not
mocked: a mock of them would only assert the shape of a call this module
makes, not that the parsing below is right, which is where the real logic is.
"""

from __future__ import annotations

from pyutilz.dev.ci_log_analyzer import CIReport, _message_prefix, analyze_log_text

_TS = "2026-08-21T20:27:50.7200744Z "


def _report() -> CIReport:
    return CIReport(run_id="123", repo="owner/repo")


class TestFailureParsing:
    def test_failed_line_is_captured_with_reason(self):
        rep = _report()
        analyze_log_text(f"{_TS}FAILED tests/test_x.py::test_a - AssertionError: boom", "j1", "pytest 3.12", rep)
        assert list(rep.failures) == ["tests/test_x.py::test_a"]
        entry = rep.failures["tests/test_x.py::test_a"]
        assert entry.kind == "FAILED"
        assert entry.reason == "AssertionError: boom"
        assert entry.jobs == {"j1"}

    def test_error_lines_are_captured_too(self):
        """A collection/fixture ERROR is a distinct kind but the same aggregation shape --
        grepping only for FAILED is exactly how setup errors get missed."""
        rep = _report()
        analyze_log_text(f"{_TS}ERROR tests/test_y.py::test_b - fixture 'db' not found", "j1", "pytest 3.12", rep)
        assert rep.failures["tests/test_y.py::test_b"].kind == "ERROR"

    def test_reasonless_failure_line_still_recorded(self):
        rep = _report()
        analyze_log_text(f"{_TS}FAILED tests/test_x.py::test_a", "j1", "n", rep)
        assert rep.failures["tests/test_x.py::test_a"].reason == ""

    def test_same_failure_across_jobs_dedupes_into_one_entry(self):
        """The whole point of the tool: one distinct failure, every job it hit."""
        rep = _report()
        line = f"{_TS}FAILED tests/test_x.py::test_a - AssertionError: boom"
        analyze_log_text(line, "j1", "pytest 3.11", rep)
        analyze_log_text(line, "j2", "pytest 3.12", rep)
        assert len(rep.failures) == 1
        assert rep.failures["tests/test_x.py::test_a"].jobs == {"j1", "j2"}

    def test_reason_backfilled_from_a_later_job(self):
        """A shard can print the node id with no reason (e.g. --tb=no); a sibling shard's
        reason must not be discarded just because the reasonless line was seen first."""
        rep = _report()
        analyze_log_text(f"{_TS}FAILED tests/test_x.py::test_a", "j1", "n1", rep)
        analyze_log_text(f"{_TS}FAILED tests/test_x.py::test_a - ValueError: real cause", "j2", "n2", rep)
        assert rep.failures["tests/test_x.py::test_a"].reason == "ValueError: real cause"

    def test_unrelated_lines_are_ignored(self):
        rep = _report()
        analyze_log_text(f"{_TS}All checks passed!\n{_TS}=== 5 passed in 1.2s ===", "j1", "n", rep)
        assert rep.failures == {}
        assert rep.warnings == {}


class TestWarningGrouping:
    _W = "  /path/to/mod.py:42: DeprecationWarning: some feature is deprecated and slated for removal"

    def test_warning_is_grouped_and_counted(self):
        rep = _report()
        analyze_log_text(f"{_TS}{self._W}", "j1", "pytest 3.12", rep)
        (group,) = rep.warnings.values()
        assert group.category == "DeprecationWarning"
        assert group.count == 1
        assert group.example_path == "/path/to/mod.py:42"

    def test_same_warning_from_different_call_sites_collapses(self):
        """Different file/line, same category+message -> ONE group with count 2. Without the
        path being dropped from the key, a 66-shard matrix reports the same warning dozens
        of times."""
        rep = _report()
        analyze_log_text(f"{_TS}  /a/one.py:1: DeprecationWarning: same text here", "j1", "n1", rep)
        analyze_log_text(f"{_TS}  /b/two.py:99: DeprecationWarning: same text here", "j2", "n2", rep)
        assert len(rep.warnings) == 1
        (group,) = rep.warnings.values()
        assert group.count == 2
        assert group.jobs == {"j1", "j2"}
        assert group.example_path == "/a/one.py:1", "first sighting is kept as the example"

    def test_different_categories_stay_separate(self):
        rep = _report()
        analyze_log_text(f"{_TS}  /a.py:1: DeprecationWarning: text", "j1", "n", rep)
        analyze_log_text(f"{_TS}  /a.py:1: UserWarning: text", "j1", "n", rep)
        assert len(rep.warnings) == 2

    def test_long_messages_differing_past_the_prefix_still_group(self):
        """Grouping on the first N words is what makes interpolated values (counts, paths,
        durations) in the tail collapse instead of splintering into one group each."""
        rep = _report()
        head = "the cache exceeded its configured budget of some number of bytes while running"
        analyze_log_text(f"{_TS}  /a.py:1: ResourceWarning: {head} 1234 bytes", "j1", "n", rep)
        analyze_log_text(f"{_TS}  /a.py:2: ResourceWarning: {head} 9999 bytes", "j1", "n", rep)
        assert len(rep.warnings) == 1

    def test_message_prefix_truncates_to_word_count(self):
        assert _message_prefix("a b c d", words=2) == "a b"
        assert _message_prefix("a b", words=5) == "a b"
        assert _message_prefix("   spaced    out   text ", words=2) == "spaced out"


class TestTimestampStripping:
    def test_line_without_timestamp_parses_identically(self):
        """Logs fetched through different paths may or may not carry the GH timestamp
        prefix; both shapes must parse."""
        with_ts, without_ts = _report(), _report()
        analyze_log_text(f"{_TS}FAILED t.py::a - boom", "j1", "n", with_ts)
        analyze_log_text("FAILED t.py::a - boom", "j1", "n", without_ts)
        assert list(with_ts.failures) == list(without_ts.failures) == ["t.py::a"]
        assert with_ts.failures["t.py::a"].reason == without_ts.failures["t.py::a"].reason


class TestRender:
    def test_render_lists_failures_job_names_and_warning_counts(self):
        rep = _report()
        analyze_log_text(f"{_TS}FAILED t.py::a - boom", "j1", "pytest 3.11", rep)
        analyze_log_text(f"{_TS}FAILED t.py::a - boom", "j2", "pytest 3.12", rep)
        analyze_log_text(f"{_TS}  /m.py:7: UserWarning: careful now", "j1", "pytest 3.11", rep)
        out = rep.render()
        assert "owner/repo run 123" in out
        assert "t.py::a" in out and "boom" in out
        assert "pytest 3.11" in out and "pytest 3.12" in out, "job NAMES, not raw ids"
        assert "2 job(s)" in out
        assert "UserWarning" in out

    def test_render_on_a_clean_run_says_none_rather_than_being_blank(self):
        out = _report().render()
        assert "(none)" in out
        assert "Failures (0 distinct)" in out

    def test_render_surfaces_unfetchable_job_logs(self):
        """A job whose log could not be fetched (still running, or past GitHub's retention
        window) must be reported, not silently counted as 'no failures'."""
        rep = _report()
        rep.fetch_errors.append("j9")
        out = rep.render()
        assert "could not fetch logs" in out and "j9" in out

    def test_render_top_n_caps_the_warning_list(self):
        rep = _report()
        for i in range(5):
            analyze_log_text(f"{_TS}  /m.py:{i}: UserWarning: distinct message number {i}", "j1", "n", rep)
        assert len(rep.warnings) == 5
        assert rep.render(top=2).count("[UserWarning]") == 2

    def test_render_orders_warnings_by_total_occurrences(self):
        rep = _report()
        analyze_log_text(f"{_TS}  /m.py:1: UserWarning: rare one", "j1", "n", rep)
        for _ in range(3):
            analyze_log_text(f"{_TS}  /m.py:2: UserWarning: common one", "j1", "n", rep)
        out = rep.render()
        assert out.index("common one") < out.index("rare one")

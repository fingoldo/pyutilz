"""Behaviour of the meta-test helpers in ``pyutilz.dev.meta_test_utils`` that are not covered by the round-trip
file: the class-name heuristics the model_dump splat detector relies on, the debt-marker scanner (excluded
fragments and unreadable files), and the findings ratchet, which reports new and drained findings against a
committed baseline and treats a missing baseline as empty."""

from __future__ import annotations

import json

from pyutilz.dev import meta_test_utils as mtu


class TestSnakeCaseVariants:
    def test_a_config_suffix_yields_the_full_name_the_stripped_name_and_the_last_word_form(self):
        assert mtu.snake_case_variants_of("TrainingBehaviorConfig") == {
            "training_behavior_config",
            "training_behavior",
            "behavior_config",
        }

    def test_a_config_suffix_inside_the_name_is_kept(self):
        assert "my_config_manager" in mtu.snake_case_variants_of("MyConfigManagerConfig")

    def test_a_name_without_a_config_suffix_is_its_own_snake_form(self):
        assert mtu.snake_case_variants_of("Plain") == {"plain", "plain_config"}


class TestScanTodoMarkers:
    def test_a_marker_comment_is_reported_with_its_location_and_keyword(self, tmp_path):
        (tmp_path / "mod.py").write_text("x = 1\n# TODO: revisit the cache\ny = 2\n", encoding="utf-8")
        hits = mtu.scan_todo_markers(tmp_path)
        assert [(path.name, lineno, kw) for path, lineno, kw, _ in hits] == [("mod.py", 2, "TODO")]

    def test_a_file_under_an_excluded_fragment_is_skipped(self, tmp_path):
        vendored = tmp_path / "vendored"
        vendored.mkdir()
        (vendored / "lib.py").write_text("# FIXME: upstream bug\n", encoding="utf-8")
        assert mtu.scan_todo_markers(tmp_path, extra_excludes=("vendored",)) == []

    def test_a_clean_tree_yields_nothing(self, tmp_path):
        (tmp_path / "mod.py").write_text("x = 1\n", encoding="utf-8")
        assert mtu.scan_todo_markers(tmp_path) == []

    def test_an_unreadable_file_is_skipped_without_aborting_the_scan(self, tmp_path):
        (tmp_path / "bad.py").write_bytes(b"\xff\xfe\x00# TODO\n")
        (tmp_path / "good.py").write_text("# HACK: works around the parser\n", encoding="utf-8")
        hits = mtu.scan_todo_markers(tmp_path)
        assert [path.name for path, *_ in hits] == ["good.py"]


class TestFindingsRatchet:
    def test_a_missing_baseline_reports_every_finding_as_new(self, tmp_path):
        new, drained = mtu.findings_ratchet(["a", "b"], tmp_path / "absent.json")
        assert new == ["a", "b"]
        assert drained == []

    def test_new_and_drained_findings_are_split_against_the_baseline(self, tmp_path):
        baseline = tmp_path / "baseline.json"
        baseline.write_text(json.dumps(["keep", "fixed"]), encoding="utf-8")
        new, drained = mtu.findings_ratchet(["keep", "added"], baseline)
        assert new == ["added"]
        assert drained == ["fixed"]

    def test_an_unchanged_set_yields_no_drift(self, tmp_path):
        baseline = tmp_path / "baseline.json"
        baseline.write_text(json.dumps(["x"]), encoding="utf-8")
        assert mtu.findings_ratchet(["x"], baseline) == ([], [])

"""Unit tests for the field-harvesting round-trip harness in\n``pyutilz.dev.meta_test_utils`` (``sentinel_for_type`` /\n``optional_scalar_fields`` / ``assert_fields_roundtrip``).\n\nThese three helpers exist to let a project write "does every optional
field on this verdict/DTO dataclass actually survive a parse round-trip"\ntests without hand-rolling per-type sentinel logic each time.\n"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional

import pytest

from pyutilz.dev.meta_test_utils import (
    assert_fields_roundtrip,
    optional_scalar_fields,
    sentinel_for_type,
)

# ---- sentinel_for_type ----------------------------------------------------

# `X | None` (PEP 604) is a runtime expression here, not an annotation --
# `from __future__ import annotations` doesn't help, so it still needs
# python>=3.10 to evaluate. Same for PEP 585 `list[str]`/`dict[str, str]`
# subscripting below, which needs python>=3.9. Build these cases
# conditionally so the file still collects on the 3.8/3.9 CI legs.
_SCALAR_CASES = [
    (bool, True),
    (Optional[bool], True),
    (str, "__sentinel_str__"),
    (Optional[str], "__sentinel_str__"),
    (float, 0.5),
    (int, 7),
]
if sys.version_info >= (3, 10):
    _SCALAR_CASES += [
        (bool | None, True),
        (str | None, "__sentinel_str__"),
        (float | None, 0.5),
        (int | None, 7),
    ]


@pytest.mark.parametrize("tp,expected", _SCALAR_CASES)
def test_sentinel_for_scalar_types(tp, expected):
    assert sentinel_for_type(tp) == expected


_NON_SCALAR_CASES = [list, dict, type(None)]
if sys.version_info >= (3, 9):
    _NON_SCALAR_CASES += [list[str], dict[str, str], Optional[list[str]]]


@pytest.mark.parametrize("tp", _NON_SCALAR_CASES)
def test_sentinel_for_non_scalar_types_returns_none(tp):
    assert sentinel_for_type(tp) is None


def test_bool_precedence_over_int():
    """bool is a subclass of int at runtime, but the annotation-object\n    identity check here must not let ``int`` accidentally win for a\n    ``bool``-annotated field."""
    assert sentinel_for_type(bool) is True
    assert sentinel_for_type(int) == 7
    assert sentinel_for_type(bool) is not sentinel_for_type(int)


# ---- optional_scalar_fields ------------------------------------------------


@dataclass
class _FakeVerdict:
    # Optional[X] rather than `X | None` -- typing.get_type_hints() evals the
    # (PEP 563 lazy) string annotation, which needs python>=3.10 for `X | None`
    # to be a valid runtime expression; Optional[X] exercises the identical
    # code path in optional_scalar_fields/sentinel_for_type down to python 3.8.
    lemma: str
    status: str
    reasoning: str
    confidence: Optional[float] = None
    mwe_type_correct: Optional[bool] = None
    corrected_label: Optional[str] = None
    priority: Optional[int] = None
    existing_synset_ids: Optional[list] = None  # non-scalar -- must be skipped
    metadata: Optional[dict] = None  # non-scalar -- must be skipped


def test_optional_scalar_fields_covers_every_scalar_field():
    sentinels = optional_scalar_fields(_FakeVerdict, skip=("lemma", "status", "reasoning"))
    assert set(sentinels) == {"confidence", "mwe_type_correct", "corrected_label", "priority"}
    assert sentinels["confidence"] == 0.5
    assert sentinels["mwe_type_correct"] is True
    assert sentinels["corrected_label"] == "__sentinel_str__"
    assert sentinels["priority"] == 7


def test_optional_scalar_fields_skips_non_scalar_fields():
    sentinels = optional_scalar_fields(_FakeVerdict)
    assert "existing_synset_ids" not in sentinels
    assert "metadata" not in sentinels


def test_optional_scalar_fields_respects_skip():
    sentinels = optional_scalar_fields(_FakeVerdict, skip=("confidence",))
    assert "confidence" not in sentinels
    assert "mwe_type_correct" in sentinels


# ---- assert_fields_roundtrip ------------------------------------------------


def test_assert_fields_roundtrip_all_match_returns_empty():
    sentinels = {"a": True, "b": "x", "c": 1.5}
    values = {"a": True, "b": "x", "c": 1.5}
    assert assert_fields_roundtrip(sentinels, lambda name: values[name]) == []


def test_assert_fields_roundtrip_detects_mismatch():
    sentinels = {"a": True, "b": "x"}
    values = {"a": True, "b": "WRONG"}
    mismatches = assert_fields_roundtrip(sentinels, lambda name: values[name])
    assert mismatches == ["b"]


def test_assert_fields_roundtrip_detects_dropped_field():
    """The classic bug: the parser never harvested the field, so the\n    parsed object's attribute is still its default (None), not the\n    sentinel the payload carried."""
    sentinels = {"mwe_type_correct": True}
    parsed = _FakeVerdict(lemma="x", status="confirmed", reasoning="")
    mismatches = assert_fields_roundtrip(sentinels, lambda name: getattr(parsed, name))
    assert mismatches == ["mwe_type_correct"]


def test_assert_fields_roundtrip_end_to_end_with_dataclass():
    parsed = _FakeVerdict(
        lemma="x", status="confirmed", reasoning="",
        confidence=0.5, mwe_type_correct=True, corrected_label="__sentinel_str__", priority=7,
    )
    sentinels = optional_scalar_fields(_FakeVerdict, skip=("lemma", "status", "reasoning"))
    mismatches = assert_fields_roundtrip(sentinels, lambda name: getattr(parsed, name))
    assert mismatches == []


# ---------------------------------------------------------------------------
# findings_ratchet / snake_case_variants_of / unbacked_audit_dispositions --
# all three are public helpers of the shared meta-test harness pyutilz exports to its
# downstream consumers, and all three were previously never mentioned anywhere under
# tests/ (audit F20, 2026-09-02).
# ---------------------------------------------------------------------------


class TestFindingsRatchet:
    def test_missing_baseline_is_treated_as_empty_so_the_first_run_reports_everything(self, tmp_path):
        from pyutilz.dev.meta_test_utils import findings_ratchet

        new, drained = findings_ratchet(["a.py:1", "b.py:2"], tmp_path / "absent.json")
        assert new == ["a.py:1", "b.py:2"]
        assert drained == []

    def test_baselined_findings_are_neither_new_nor_drained(self, tmp_path):
        import json

        from pyutilz.dev.meta_test_utils import findings_ratchet

        baseline = tmp_path / "b.json"
        baseline.write_text(json.dumps(["a.py:1", "b.py:2"]), encoding="utf-8")

        new, drained = findings_ratchet(["a.py:1", "b.py:2"], baseline)
        assert new == [] and drained == []

    def test_reports_new_and_drained_separately_and_sorted(self, tmp_path):
        import json

        from pyutilz.dev.meta_test_utils import findings_ratchet

        baseline = tmp_path / "b.json"
        baseline.write_text(json.dumps(["old2.py:2", "kept.py:9", "old1.py:1"]), encoding="utf-8")

        new, drained = findings_ratchet(["kept.py:9", "z_new.py:5", "a_new.py:3"], baseline)
        assert new == ["a_new.py:3", "z_new.py:5"]
        assert drained == ["old1.py:1", "old2.py:2"]

    def test_duplicate_findings_are_deduplicated(self, tmp_path):
        from pyutilz.dev.meta_test_utils import findings_ratchet

        new, _ = findings_ratchet(["dup.py:1", "dup.py:1"], tmp_path / "absent.json")
        assert new == ["dup.py:1"]

    def test_the_baseline_file_is_never_written_by_the_ratchet_itself(self, tmp_path):
        """It is a pure diff: refreshing is the caller's explicit, reviewed action."""
        from pyutilz.dev.meta_test_utils import findings_ratchet

        baseline = tmp_path / "absent.json"
        findings_ratchet(["a.py:1"], baseline)
        assert not baseline.exists()


class TestSnakeCaseVariantsOf:
    def test_documented_three_way_split(self):
        from pyutilz.dev.meta_test_utils import snake_case_variants_of

        assert snake_case_variants_of("TrainingBehaviorConfig") >= {
            "training_behavior_config",
            "training_behavior",
            "behavior_config",
        }

    def test_single_word_class_yields_its_lowercase_form(self):
        from pyutilz.dev.meta_test_utils import snake_case_variants_of

        assert "config" in snake_case_variants_of("Config")

    def test_two_word_class(self):
        from pyutilz.dev.meta_test_utils import snake_case_variants_of

        assert "model_config" in snake_case_variants_of("ModelConfig")

    def test_result_is_a_set_of_lowercase_snake_case_names(self):
        from pyutilz.dev.meta_test_utils import snake_case_variants_of

        out = snake_case_variants_of("ModelHyperparamsConfig")
        assert isinstance(out, set)
        assert out
        assert all(n == n.lower() and " " not in n and "-" not in n for n in out)


class TestUnbackedAuditDispositions:
    @staticmethod
    def _write(base, rel, text):
        path = base / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_a_resolved_row_citing_an_existing_file_is_backed(self, tmp_path):
        from pyutilz.dev.meta_test_utils import unbacked_audit_dispositions

        repo = tmp_path / "repo"
        self._write(repo, "src/mod.py", "x = 1\n")
        audit = repo / "audits"
        self._write(audit, "a.md", "| F01 | RESOLVED | fixed in `src/mod.py` |\n")

        assert unbacked_audit_dispositions(audit, repo) == []

    def test_a_resolved_row_citing_an_existing_test_function_is_backed(self, tmp_path):
        from pyutilz.dev.meta_test_utils import unbacked_audit_dispositions

        repo = tmp_path / "repo"
        self._write(repo, "tests/test_thing.py", "def test_it_works():\n    assert True\n")
        audit = repo / "audits"
        self._write(audit, "a.md", "| F01 | RESOLVED | covered by `test_it_works` |\n")

        assert unbacked_audit_dispositions(audit, repo) == []

    def test_a_resolved_row_citing_nothing_is_reported(self, tmp_path):
        from pyutilz.dev.meta_test_utils import unbacked_audit_dispositions

        repo = tmp_path / "repo"
        self._write(repo, "tests/test_thing.py", "def test_it_works():\n    assert True\n")
        audit = repo / "audits"
        self._write(audit, "a.md", "| F02 | RESOLVED | trust me |\n")

        out = unbacked_audit_dispositions(audit, repo)
        assert len(out) == 1
        assert "a.md:1" in out[0] and "cites nothing" in out[0]

    def test_a_resolved_row_citing_a_vanished_path_is_reported(self, tmp_path):
        """The exact failure mode this rule exists for: a citation that used to resolve."""
        from pyutilz.dev.meta_test_utils import unbacked_audit_dispositions

        repo = tmp_path / "repo"
        self._write(repo, "tests/test_thing.py", "def test_it_works():\n    assert True\n")
        audit = repo / "audits"
        self._write(audit, "a.md", "| F03 | RESOLVED | see `src/moved_away.py` |\n")

        assert len(unbacked_audit_dispositions(audit, repo)) == 1

    def test_prose_saying_resolved_in_lower_case_is_not_a_disposition(self, tmp_path):
        """The marker is the fixed upper-case vocabulary word, not the English verb. Matching it case-
        insensitively claimed 163 ordinary table rows on autopsia 2026-09-03 - "...the parameters were
        resolved at age 70..." among them - each arriving as a demand to cite an artefact for a sentence
        that was never a disposition."""
        from pyutilz.dev.meta_test_utils import unbacked_audit_dispositions

        repo = tmp_path / "repo"
        self._write(repo, "src/mod.py", "x = 1\n")
        audit = repo / "audits"
        self._write(audit, "a.md", "| F01 | P1 | the parameters were resolved at age 70 while the gate ignored them |\n")

        assert unbacked_audit_dispositions(audit, repo) == []

    def test_a_field_name_containing_resolved_is_not_a_disposition(self, tmp_path):
        """The same failure with no English verb in sight: the row's only hit was a serialised field name."""
        from pyutilz.dev.meta_test_utils import unbacked_audit_dispositions

        repo = tmp_path / "repo"
        self._write(repo, "src/mod.py", "x = 1\n")
        audit = repo / "audits"
        self._write(audit, "a.md", "| F02 | P2 | `resolved[].observation_kind` is stripped by the response model |\n")

        assert unbacked_audit_dispositions(audit, repo) == []

    def test_a_resolved_row_citing_a_symbol_written_as_a_call_is_backed(self, tmp_path):
        """A citation names a function the way prose names one - with the parentheses, or qualified by the
        class it hangs off. Requiring an exact string match reported 25 autopsia rows citing real, findable
        symbols as citing nothing."""
        from pyutilz.dev.meta_test_utils import unbacked_audit_dispositions

        repo = tmp_path / "repo"
        self._write(repo, "src/obs.py", "class Observation:\n    def is_all_population(self):\n        return True\n")
        audit = repo / "audits"
        self._write(audit, "call.md", "| F03 | RESOLVED | fixed in `Observation.is_all_population()` |\n")
        self._write(audit, "bare.md", "| F04 | RESOLVED | fixed in `is_all_population()` |\n")

        assert unbacked_audit_dispositions(audit, repo) == []

    def test_a_resolved_row_merely_mentioning_an_identifier_is_still_reported(self, tmp_path):
        """The other half of the bar. Accepting any row that CONTAINS a real name would make this rule fire
        on nothing, so a citation has to BE a name rather than merely mention one."""
        from pyutilz.dev.meta_test_utils import unbacked_audit_dispositions

        repo = tmp_path / "repo"
        self._write(repo, "src/mod.py", "def differential():\n    return 1\n")
        audit = repo / "audits"
        self._write(audit, "a.md", '| F05 | RESOLVED | see `assert differential[0]["rank"] == 1` |\n')

        assert len(unbacked_audit_dispositions(audit, repo)) == 1

    def test_rows_that_are_not_resolved_are_ignored(self, tmp_path):
        from pyutilz.dev.meta_test_utils import unbacked_audit_dispositions

        repo = tmp_path / "repo"
        self._write(repo, "tests/test_thing.py", "def test_it_works():\n    assert True\n")
        audit = repo / "audits"
        self._write(audit, "a.md", "| F04 | OPEN | nothing done yet |\n| F05 | WONTFIX | declined |\n")

        assert unbacked_audit_dispositions(audit, repo) == []

    def test_non_table_lines_are_ignored(self, tmp_path):
        from pyutilz.dev.meta_test_utils import unbacked_audit_dispositions

        repo = tmp_path / "repo"
        self._write(repo, "tests/test_thing.py", "def test_it_works():\n    assert True\n")
        audit = repo / "audits"
        self._write(audit, "a.md", "This finding was RESOLVED last week, honest.\n")

        assert unbacked_audit_dispositions(audit, repo) == []

    def test_an_unresolved_row_is_not_treated_as_resolved(self, tmp_path):
        """Regression, product bug fixed 2026-09-02 in meta_test_utils.py:619.

        ``_DISPOSITION_RE`` was ``re.compile("RESOLVED", re.IGNORECASE)`` with no word boundary,
        so the substring match fired on UNRESOLVED (and on PARTIALLY-UNRESOLVED, "not resolved",
        ...). Every row a maintainer had honestly marked UNRESOLVED was then demanded to cite an
        artefact and reported as an unbacked RESOLVED claim -- noise pointing at exactly the rows
        that were already correctly flagged as outstanding work. The regex now anchors on
        ``\bRESOLVED\b``.
        """
        from pyutilz.dev.meta_test_utils import unbacked_audit_dispositions

        repo = tmp_path / "repo"
        self._write(repo, "tests/test_thing.py", "def test_it_works():\n    assert True\n")
        audit = repo / "audits"
        self._write(audit, "a.md", "| F06 | UNRESOLVED | still being worked on |\n")

        assert unbacked_audit_dispositions(audit, repo) == []

    def test_a_resolved_row_alongside_an_unresolved_one_is_still_checked(self, tmp_path):
        """The word-boundary fix must not make the scanner blind to genuine RESOLVED rows."""
        from pyutilz.dev.meta_test_utils import unbacked_audit_dispositions

        repo = tmp_path / "repo"
        self._write(repo, "tests/test_thing.py", "def test_it_works():\n    assert True\n")
        audit = repo / "audits"
        self._write(audit, "a.md", "| F07 | UNRESOLVED | pending |\n| F08 | RESOLVED | trust me |\n")

        reported = unbacked_audit_dispositions(audit, repo)
        assert len(reported) == 1
        assert reported[0].endswith("RESOLVED cites nothing")

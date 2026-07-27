"""The runaway-decoder check: it must flag repetition and must NOT flag an honest verbose extraction."""

from __future__ import annotations

from pyutilz.llm.degeneracy import DegeneracyThresholds, degeneracy_report


def test_an_honest_extraction_is_not_flagged_and_the_report_says_what_it_measured() -> None:
    """A checker that reports only its hits cannot be told from one that never ran, so the counts travel."""
    honest = [((f"c{i}", "CAUSES", f"f{i}"), f"quote {i}") for i in range(12)]
    r = degeneracy_report(honest)
    assert r.degenerate is False
    assert r.n_records == 12 and r.duplicate_records == 0
    assert r.thresholds["max_records"] == DegeneracyThresholds().max_records


def test_the_three_shapes_a_repeating_decoder_leaves_behind_are_each_caught() -> None:
    """Identical records, one span reused across unrelated records, and an implausible count."""
    same_claim = [(("hanging", "CAUSES", "death"), f"q{i}") for i in range(10)]
    assert degeneracy_report(same_claim).degenerate is True

    one_span = [((f"s{i}", "CAUSES", f"o{i}"), "the same sentence") for i in range(12)]
    assert any("one_quote_reused" in f for f in degeneracy_report(one_span).flags)

    runaway = [((f"s{i}", "CAUSES", f"o{i}"), f"q{i}") for i in range(231)]
    assert any("record_count_implausible" in f for f in degeneracy_report(runaway).flags)


def test_re_wrapped_copies_of_one_span_count_as_one() -> None:
    """A decoder re-wrapping the sentence it is copying must not thereby escape the reuse count."""
    wrapped = [((f"s{i}", "CAUSES", f"o{i}"), "the  same\n sentence " if i % 2 else "the same sentence") for i in range(12)]
    assert any("one_quote_reused" in f for f in degeneracy_report(wrapped).flags)


def test_a_caller_may_raise_the_bound_without_silencing_the_check() -> None:
    """Documents that legitimately yield hundreds of records raise the bound; they do not disable it."""
    many = [((f"s{i}", "CAUSES", f"o{i}"), f"q{i}") for i in range(200)]
    assert degeneracy_report(many).degenerate is True
    generous = degeneracy_report(many, DegeneracyThresholds(max_records=500))
    assert generous.degenerate is False and generous.n_records == 200

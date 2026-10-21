"""The tests split out of test_audit_20260903_scanner_fixes_a_l.py that touch pyutilz-only code (field_text_agreement, the CLI, the pyutilz facade); the rest moved to py-ci-shared."""

from __future__ import annotations


def test_field_text_agreement_rejects_a_cue_that_normalises_to_nothing():
    """F191: such a cue compiled to a double word boundary and matched every record."""
    from pyutilz.dev.code_audit.field_text_agreement import FieldTextRule, cues_in_text

    rule = FieldTextRule(name="x", field="f", text_fields=("t",), cues={"postmortem": ["_"]})
    assert cues_in_text(rule, "vital hanging") == {}
    rule = FieldTextRule(name="x", field="f", text_fields=("t",), cues={"postmortem": ["postmortem"]})
    assert cues_in_text(rule, "a post-mortem was held") == {"postmortem": "postmortem"}


def test_field_text_agreement_tiebreak_prefers_the_longest_cue():
    """F197: the alphabetically-first value won, so a rename changed the verdict."""
    from pyutilz.dev.code_audit.field_text_agreement import FieldTextRule, check_record

    rule = FieldTextRule(name="x", field="f", text_fields=("t",), cues={"alpha": ["one"], "zeta": ["a much longer phrase"]})
    verdict = check_record(rule, {"f": "", "t": "a much longer phrase and one"})
    assert verdict.supported == "zeta"
    assert verdict.alternatives == ("alpha",)

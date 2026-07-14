"""Unit tests for the field-harvesting round-trip harness in
``pyutilz.dev.meta_test_utils`` (``sentinel_for_type`` /
``optional_scalar_fields`` / ``assert_fields_roundtrip``).

These three helpers exist to let a project write "does every optional
field on this verdict/DTO dataclass actually survive a parse round-trip"
tests without hand-rolling per-type sentinel logic each time.
"""

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
    """bool is a subclass of int at runtime, but the annotation-object
    identity check here must not let ``int`` accidentally win for a
    ``bool``-annotated field."""
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
    """The classic bug: the parser never harvested the field, so the
    parsed object's attribute is still its default (None), not the
    sentinel the payload carried."""
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

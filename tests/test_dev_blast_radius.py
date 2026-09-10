"""pyutilz.dev.blast_radius: a write may change its target rows and declared columns, and nothing else.

Each case is one of the glossum defects the persistence gates passed, rebuilt on a tiny store.
"""

from __future__ import annotations

import asyncio
import copy
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from pyutilz.dev.blast_radius import assert_only_target_changed, blast_radius_violations, diff_snapshots, records_snapshot, sql_snapshot

T = "generated_cognates"


def _store() -> Dict[str, List[Dict[str, Any]]]:
    """Two cognates of one lemma: the judged one and its sibling. The sibling is what a looser WHERE overwrites."""
    return {
        T: [
            {"id": 1, "lemma": "слово", "cognate": "slovo", "validation_status_id": None, "producer_note": "from translate", "aspect_pair_correct": None},
            {"id": 2, "lemma": "слово", "cognate": "slava", "validation_status_id": None, "producer_note": "from translate", "aspect_pair_correct": None},
        ]
    }


def _snap(store: Dict[str, List[Dict[str, Any]]]):
    return records_snapshot(store, {T: ["id"]})


DECLARED = {"targets": {T: [(1,)]}, "allowed_columns": {T: ["validation_status_id", "aspect_pair_correct"]}}


class TestTheDefectsThePersistenceGatesMissed:
    def test_a_verdict_that_reaches_a_sibling_row_fails(self) -> None:
        """N-9 / N-16: ``a AND b AND c OR d`` wrote one verdict onto every cognate of the lemma."""
        store = _store()
        before = _snap(store)
        for row in store[T]:
            row["validation_status_id"] = 4

        with pytest.raises(AssertionError, match=r"did not target changed \['validation_status_id'\]"):
            assert_only_target_changed(before, _snap(store), **DECLARED)

    def test_a_verdict_written_over_the_producers_column_fails(self) -> None:
        """N-18: the fix stored the verdict's reasoning in the producer's note, erasing it."""
        store = _store()
        before = _snap(store)
        store[T][0].update(validation_status_id=2, producer_note="a loan translation")

        with pytest.raises(AssertionError, match=r"outside the write's declared set changed \['producer_note'\]"):
            assert_only_target_changed(before, _snap(store), **DECLARED)

    def test_a_stored_value_reset_to_null_fails(self) -> None:
        """N-10: aspect_pair_correct=False was stored, then set back to NULL a few lines later."""
        store = _store()
        store[T][0]["aspect_pair_correct"] = False
        before = _snap(store)
        store[T][0].update(validation_status_id=2, aspect_pair_correct=None)

        with pytest.raises(AssertionError, match=r"became NULL \['aspect_pair_correct'\]"):
            assert_only_target_changed(before, _snap(store), **DECLARED)


class TestAWriteThatStaysInsideItsReachPasses:
    def test_the_target_row_and_declared_columns_only(self) -> None:
        store = _store()
        before = _snap(store)
        store[T][0].update(validation_status_id=2, aspect_pair_correct=False)

        changes = assert_only_target_changed(before, _snap(store), **DECLARED)

        assert [(c.table, c.key, sorted(c.columns)) for c in changes] == [(T, (1,), ["aspect_pair_correct", "validation_status_id"])]

    def test_a_declared_clear_is_allowed(self) -> None:
        store = _store()
        store[T][0]["aspect_pair_correct"] = True
        before = _snap(store)
        store[T][0]["aspect_pair_correct"] = None

        assert_only_target_changed(before, _snap(store), **DECLARED, may_clear={T: ["aspect_pair_correct"]})

    def test_an_ignored_column_is_not_a_change(self) -> None:
        store = _store()
        for row in store[T]:
            row["updated_at"] = "t0"
        before = _snap(store)
        store[T][0].update(validation_status_id=2)
        store[T][1]["updated_at"] = "t1"

        assert_only_target_changed(before, _snap(store), **DECLARED, ignore_columns=["updated_at"])


class TestTheCheckCannotPassByDoingNothing:
    def test_a_write_that_changed_nothing_fails(self) -> None:
        store = _store()

        with pytest.raises(AssertionError, match="changed nothing at all"):
            assert_only_target_changed(_snap(store), _snap(store), **DECLARED)

    def test_an_undeclared_insert_or_delete_fails(self) -> None:
        store = _store()
        before = _snap(store)
        after_store = copy.deepcopy(store)
        after_store[T].append({"id": 3, "lemma": "x"})
        del after_store[T][1]

        problems = blast_radius_violations(diff_snapshots(before, _snap(after_store)), **DECLARED)

        assert any("inserted" in p for p in problems) and any("deleted" in p for p in problems)


class TestTheSqlSnapshot:
    def test_rows_are_keyed_by_the_given_columns(self) -> None:
        class _Result:
            def __init__(self, rows: List[Dict[str, Any]]) -> None:
                self.rows = rows

            def mappings(self) -> "_Result":
                return self

            def all(self) -> List[Dict[str, Any]]:
                return self.rows

        session = MagicMock()
        session.execute = AsyncMock(return_value=_Result([{"sense_id": "s1", "lang": 2, "v": None}, {"sense_id": "s2", "lang": 2, "v": 1}]))

        snap = asyncio.run(sql_snapshot(session, {"llm.flags": ["sense_id", "lang"]}, where={"llm.flags": "lang = 2"}))

        assert set(snap["llm.flags"]) == {("s1", 2), ("s2", 2)}
        session.execute.assert_awaited_once()
        assert str(session.execute.call_args.args[0]) == 'SELECT * FROM "llm"."flags" WHERE lang = 2'

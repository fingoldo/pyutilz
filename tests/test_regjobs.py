"""Behaviour of ``pyutilz.database.db.regjobs``: each helper sends its statement through the facade with the job name
and the column it touches (heartbeats stamp ``last_ping_at`` and ``last_result``, finalize stamps ``finished_at`` and
``result``), dict results are wrapped in the jsonb adapter while scalars pass through, and the table name defaults to
``regular_jobs``. The facade's ``safe_execute`` is replaced by a recorder, so no Postgres server is needed."""

from __future__ import annotations

import pytest

import pyutilz.database.db as facade
from pyutilz.database.db import regjobs
from pyutilz.database.db._common import Json


@pytest.fixture
def executed(monkeypatch):
    calls: list[tuple[object, object]] = []

    def record(query, params=None, *args, **kwargs):
        calls.append((query, params))
        return "done"

    monkeypatch.setattr(facade, "safe_execute", record)
    return calls


class TestCreateTable:
    def test_the_default_table_name_is_regular_jobs(self, executed):
        regjobs.regjobs_create_table()
        assert len(executed) == 1
        assert "regular_jobs" in repr(executed[0][0])

    def test_a_custom_table_name_is_used_for_the_table(self, executed):
        regjobs.regjobs_create_table("my_jobs")
        assert "my_jobs" in repr(executed[0][0])


class TestPoll:
    def test_the_job_and_the_claimant_are_passed_as_parameters(self, executed):
        assert regjobs.regjobs_poll("nightly", "node-7") == "done"
        assert executed[0][1] == {"job_name": "nightly", "taken_by": "node-7"}

    def test_the_claim_locks_the_row_and_stamps_the_start(self, executed):
        regjobs.regjobs_poll("nightly", "node-7")
        text = repr(executed[0][0])
        assert "for update skip locked" in text
        assert "taken_by" in text and "started_at" in text


class TestProgress:
    def test_a_heartbeat_stamps_the_ping_and_stores_the_interim_result(self, executed):
        regjobs.regjobs_progress("nightly", {"done": 3})
        query, params = executed[0]
        assert "last_ping_at" in repr(query)
        assert "last_result" in repr(query)
        assert params["job_name"] == "nightly"
        assert isinstance(params["result"], Json)

    def test_a_scalar_result_is_passed_through_unwrapped(self, executed):
        regjobs.regjobs_progress("nightly", "running")
        assert executed[0][1]["result"] == "running"


class TestFinalize:
    def test_finalize_stamps_the_finish_and_stores_the_final_result(self, executed):
        regjobs.regjobs_finalize("nightly", {"rows": 10})
        query, params = executed[0]
        assert "finished_at" in repr(query)
        assert "result" in repr(query)
        assert isinstance(params["result"], Json)
        assert params["job_name"] == "nightly"

    def test_a_list_result_is_wrapped_as_jsonb_too(self, executed):
        regjobs.regjobs_finalize("nightly", [1, 2])
        assert isinstance(executed[0][1]["result"], Json)

    def test_a_custom_table_is_threaded_through(self, executed):
        regjobs.regjobs_finalize("nightly", {}, table_name="other_jobs")
        assert "other_jobs" in repr(executed[0][0])

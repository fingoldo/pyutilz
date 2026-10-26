"""The ``gh`` CLI paths of ``pyutilz.dev.ci_log_analyzer``: JSON parsing, job filtering, log fetching,
run-level collection of fetch errors, and the command-line entry point. ``subprocess.run`` is replaced,
so these tests need no GitHub access and no ``gh`` binary."""

from __future__ import annotations

import json
import subprocess

import pytest

from pyutilz.dev import ci_log_analyzer as cla


class _Completed:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout


def _record_run(monkeypatch, stdout: str = "", error: Exception | None = None) -> list[list[str]]:
    """Replace ``subprocess.run`` with a recorder that returns ``stdout`` or raises ``error``; returns the argv log."""
    calls: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        calls.append(list(argv))
        if error is not None:
            raise error
        return _Completed(stdout)

    monkeypatch.setattr(cla.subprocess, "run", fake_run)
    return calls


class TestGhJson:
    def test_parses_the_json_stdout_of_the_gh_command(self, monkeypatch):
        calls = _record_run(monkeypatch, stdout='{"jobs": [{"databaseId": 7}]}')
        assert cla._gh_json(["run", "view", "1"]) == {"jobs": [{"databaseId": 7}]}
        assert calls == [["gh", "run", "view", "1"]]

    def test_a_non_zero_exit_propagates(self, monkeypatch):
        _record_run(monkeypatch, error=subprocess.CalledProcessError(1, ["gh"]))
        with pytest.raises(subprocess.CalledProcessError):
            cla._gh_json(["run", "list"])


class TestListJobIds:
    _JOBS = {
        "jobs": [
            {"databaseId": 11, "name": "test (3.8)", "conclusion": "failure"},
            {"databaseId": 12, "name": "lint", "conclusion": "success"},
            {"databaseId": 13, "name": "docs", "conclusion": "failure"},
        ]
    }

    def test_keeps_only_the_jobs_with_the_requested_conclusion(self, monkeypatch):
        _record_run(monkeypatch, stdout=json.dumps(self._JOBS["jobs"]))
        assert cla._list_job_ids("o/r", "99", "failure") == {"11": "test (3.8)", "13": "docs"}

    def test_none_keeps_every_job_regardless_of_conclusion(self, monkeypatch):
        _record_run(monkeypatch, stdout=json.dumps(self._JOBS["jobs"]))
        assert cla._list_job_ids("o/r", "99", None) == {"11": "test (3.8)", "12": "lint", "13": "docs"}

    def test_the_gh_query_names_the_repo_and_the_run(self, monkeypatch):
        calls = _record_run(monkeypatch, stdout=json.dumps(self._JOBS["jobs"]))
        cla._list_job_ids("fingoldo/pyutilz", "4242", None)
        assert calls == [["gh", "run", "view", "4242", "--repo", "fingoldo/pyutilz", "--json", "jobs", "--jq", ".jobs"]]


class TestFetchJobLog:
    def test_returns_the_raw_log_text_of_the_job(self, monkeypatch):
        calls = _record_run(monkeypatch, stdout="line one\nline two\n")
        assert cla._fetch_job_log("o/r", "55") == "line one\nline two\n"
        assert calls == [["gh", "api", "repos/o/r/actions/jobs/55/logs"]]

    def test_an_unavailable_log_raises(self, monkeypatch):
        _record_run(monkeypatch, error=subprocess.CalledProcessError(1, ["gh"]))
        with pytest.raises(subprocess.CalledProcessError):
            cla._fetch_job_log("o/r", "55")


class TestAnalyzeRun:
    _FAILED_LOG = "tests/test_a.py::test_b FAILED - boom\n"

    def test_collects_the_failure_from_every_failed_job(self, monkeypatch):
        jobs = {"jobs": [{"databaseId": 1, "name": "tests (3.8)", "conclusion": "failure"}]}
        _record_run(monkeypatch, stdout=json.dumps(jobs["jobs"]))
        monkeypatch.setattr(cla, "_fetch_job_log", lambda repo, job_id: self._FAILED_LOG)
        report = cla.analyze_run("o/r", "9")
        assert report.run_id == "9"
        assert report.repo == "o/r"
        assert report.job_names == {"1": "tests (3.8)"}
        assert report.fetch_errors == []

    def test_a_job_whose_log_cannot_be_fetched_is_named_in_fetch_errors(self, monkeypatch):
        jobs = {"jobs": [{"databaseId": 1, "name": "tests (3.8)", "conclusion": "failure"}]}
        _record_run(monkeypatch, stdout=json.dumps(jobs["jobs"]))

        def unavailable(repo, job_id):
            raise subprocess.CalledProcessError(1, ["gh"])

        monkeypatch.setattr(cla, "_fetch_job_log", unavailable)
        report = cla.analyze_run("o/r", "9")
        assert report.fetch_errors == ["tests (3.8) (job id 1)"]
        assert report.failures == {}

    def test_the_default_conclusion_fetches_only_failed_jobs(self, monkeypatch):
        jobs = {
            "jobs": [
                {"databaseId": 1, "name": "tests", "conclusion": "failure"},
                {"databaseId": 2, "name": "lint", "conclusion": "success"},
            ]
        }
        _record_run(monkeypatch, stdout=json.dumps(jobs["jobs"]))
        fetched: list[str] = []

        def record_fetch(repo, job_id):
            fetched.append(job_id)
            return ""

        monkeypatch.setattr(cla, "_fetch_job_log", record_fetch)
        cla.analyze_run("o/r", "9")
        assert fetched == ["1"]


class TestMain:
    def test_prints_the_rendered_report_for_the_given_run(self, monkeypatch, capsys):
        seen: dict = {}

        def fake_analyze(repo, run_id, conclusion="failure"):
            seen.update(repo=repo, run_id=run_id, conclusion=conclusion)
            return cla.CIReport(run_id=run_id, repo=repo)

        monkeypatch.setattr(cla, "analyze_run", fake_analyze)
        monkeypatch.setattr("sys.argv", ["ci_log_analyzer", "fingoldo/pyutilz", "77"])
        cla.main()
        assert seen == {"repo": "fingoldo/pyutilz", "run_id": "77", "conclusion": "failure"}
        assert capsys.readouterr().out.strip()

    def test_conclusion_all_analyzes_every_job(self, monkeypatch):
        seen: dict = {}

        def fake_analyze(repo, run_id, conclusion="failure"):
            seen["conclusion"] = conclusion
            return cla.CIReport(run_id=run_id, repo=repo)

        monkeypatch.setattr(cla, "analyze_run", fake_analyze)
        monkeypatch.setattr("sys.argv", ["ci_log_analyzer", "o/r", "5", "--conclusion", "all"])
        cla.main()
        assert seen["conclusion"] is None

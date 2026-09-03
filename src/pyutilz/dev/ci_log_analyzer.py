"""Aggregate FAILED/ERROR lines and pytest warning summaries across every job of a GitHub Actions
run, so a 66-shard matrix can be triaged from one consolidated report instead of opening each job's
log by hand.

Shells out to the ``gh`` CLI (reuses the caller's existing auth) rather than talking to the GitHub
API directly, so no token handling lives here.

CLI usage::

    python -m pyutilz.dev.ci_log_analyzer OWNER/REPO RUN_ID [--conclusion failure] [--top 30]

Library usage::

    from pyutilz.dev.ci_log_analyzer import analyze_run
    report = analyze_run("fingoldo/mlframe", "32500615829")
    print(report.render())
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess  # nosec B404 - shells out to the trusted local `gh` CLI only, no untrusted input reaches the shell
from dataclasses import dataclass, field
from typing import Any

# Strips the leading GH Actions log timestamp ("2026-08-21T20:27:50.7200744Z ") so lines compare cleanly.
_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z\s?")
# Warning lines from pytest's "warnings summary" section look like:
#   "  /path/to/file.py:123: SomeWarning: the message text"
# Collapse the path/line-number to a category key so the same warning from different call sites
# (or across shards hitting different test files) groups together; keep one example path per group.
_WARNING_RE = re.compile(r"^\s*(?P<path>\S+\.py):(?P<line>\d+):\s*(?P<category>\w+Warning):\s*(?P<message>.*)$")
_FAILED_RE = re.compile(r"^(?P<kind>FAILED|ERROR)\s+(?P<nodeid>\S+)(?:\s*-\s*(?P<reason>.*))?$")


@dataclass
class WarningGroup:
    """One pytest-warning category, collapsed across every job and call site that emitted it."""

    category: str
    message_prefix: str
    count: int = 0
    example_path: str = ""
    jobs: set[str] = field(default_factory=set)


@dataclass
class FailureEntry:
    """One failing test node id, with every job that reported it."""

    kind: str  # "FAILED" or "ERROR"
    nodeid: str
    reason: str
    jobs: set[str] = field(default_factory=set)


@dataclass
class CIReport:
    """Deduplicated failures + warning groups aggregated across all jobs of one run."""

    run_id: str
    repo: str
    failures: dict[str, FailureEntry] = field(default_factory=dict)  # nodeid -> FailureEntry
    warnings: dict[tuple[str, str], WarningGroup] = field(default_factory=dict)  # (category, message_prefix) -> WarningGroup
    job_names: dict[str, str] = field(default_factory=dict)  # job_id -> name
    fetch_errors: list[str] = field(default_factory=list)  # "<job name> (job id <id>)" for each job whose log couldn't be fetched

    def render(self, top: int = 30) -> str:
        """Human-readable summary: failures first (deduped, with every job they hit), then the
        top-N warning categories by total occurrence count across all jobs."""
        lines = [f"CI report for {self.repo} run {self.run_id}", ""]

        if self.fetch_errors:
            lines.append(f"WARNING: could not fetch logs for {len(self.fetch_errors)} job(s): {self.fetch_errors}")
            lines.append("")

        lines.append(f"=== Failures ({len(self.failures)} distinct) ===")
        if not self.failures:
            lines.append("(none)")
        for entry in sorted(self.failures.values(), key=lambda e: (-len(e.jobs), e.nodeid)):
            job_list = ", ".join(sorted(self.job_names.get(j, j) for j in entry.jobs))
            lines.append(f"[{entry.kind}] {entry.nodeid}  ({len(entry.jobs)} job(s): {job_list})")
            if entry.reason:
                lines.append(f"    {entry.reason[:300]}")
        lines.append("")

        lines.append(f"=== Top {top} warning categories by total occurrences ===")
        ranked = sorted(self.warnings.values(), key=lambda w: -w.count)
        if not ranked:
            lines.append("(none)")
        for w in ranked[:top]:
            job_list = ", ".join(sorted(self.job_names.get(j, j) for j in w.jobs))
            lines.append(f"[{w.category}] x{w.count} across {len(w.jobs)} job(s) ({job_list})")
            lines.append(f"    e.g. {w.example_path}: {w.message_prefix[:200]}")
        return "\n".join(lines)


def _gh_json(args: list[str]) -> Any:
    """Run a `gh` CLI command expecting JSON output; raises on non-zero exit.

    Returns ``Any`` deliberately: the shape is whatever the requested ``--json`` fields
    produce, so each caller annotates the concrete shape it asked for at its own use site.
    """
    # encoding="utf-8", errors="replace" -- NOT text=True, which decodes with
    # locale.getpreferredencoding(False) (cp1251 on a Russian-locale Windows box). CI logs are
    # UTF-8: a byte undefined in the ambient codepage raises UnicodeDecodeError inside subprocess's
    # own reader thread, where it is swallowed, and stdout silently becomes None -- json.loads(None)
    # then raises TypeError with no trace of the real cause. Without such a byte it merely mojibakes.
    result = subprocess.run(["gh", *args], capture_output=True, encoding="utf-8", errors="replace", check=True)  # nosec B603 B607 - fixed `gh` binary, args are caller-controlled repo/run identifiers, not untrusted input
    return json.loads(result.stdout)


def _list_job_ids(repo: str, run_id: str, conclusion: str | None) -> dict[str, str]:
    """Returns {job_id: job_name} for every job in the run, optionally filtered by conclusion
    (e.g. "failure" to skip the (usually much larger) set of passing jobs' logs)."""
    jobs: list[dict[str, Any]] = _gh_json(["run", "view", run_id, "--repo", repo, "--json", "jobs", "--jq", ".jobs"])
    out: dict[str, str] = {}
    for j in jobs:
        if conclusion is not None and j.get("conclusion") != conclusion:
            continue
        out[str(j["databaseId"])] = j["name"]
    return out


def _fetch_job_log(repo: str, job_id: str) -> str:
    """Raw log text for one job. Raises ``CalledProcessError`` when the log is unavailable --
    notably while the run is still in progress, and after GitHub's retention window expires."""
    result = subprocess.run(  # nosec B603 B607 - fixed `gh` binary, job_id/repo are caller-controlled identifiers
        ["gh", "api", f"repos/{repo}/actions/jobs/{job_id}/logs"],
        capture_output=True,
        # See _gh_json: text=True would decode UTF-8 CI logs with the ambient codepage, losing the
        # whole log to a swallowed UnicodeDecodeError.
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return result.stdout


def _message_prefix(message: str, words: int = 12) -> str:
    """First N words of a warning message, used as a grouping key so near-identical messages
    (e.g. the same warning with a different numeric value interpolated) still collapse together."""
    return " ".join(message.split()[:words])


def analyze_log_text(text: str, job_id: str, job_name: str, report: CIReport) -> None:
    """Scan one job's raw log text, updating `report` in place."""
    for raw_line in text.splitlines():
        line = _TS_RE.sub("", raw_line)

        m = _FAILED_RE.match(line)
        if m:
            nodeid = m.group("nodeid")
            entry = report.failures.setdefault(nodeid, FailureEntry(kind=m.group("kind"), nodeid=nodeid, reason=m.group("reason") or ""))
            entry.jobs.add(job_id)
            if not entry.reason and m.group("reason"):
                entry.reason = m.group("reason")
            continue

        m = _WARNING_RE.match(line)
        if m:
            prefix = _message_prefix(m.group("message"))
            key = (m.group("category"), prefix)
            group = report.warnings.setdefault(key, WarningGroup(category=m.group("category"), message_prefix=prefix))
            group.count += 1
            group.jobs.add(job_id)
            if not group.example_path:
                group.example_path = f"{m.group('path')}:{m.group('line')}"

    report.job_names[job_id] = job_name


def analyze_run(repo: str, run_id: str, conclusion: str | None = "failure") -> CIReport:
    """Fetch and analyze every job's log for one GitHub Actions run.

    Args:
        repo: "owner/repo".
        run_id: the GH Actions run ID (as shown in `gh run list` / the Actions URL).
        conclusion: only fetch logs for jobs with this conclusion (default "failure", the common
            case: passing jobs rarely need triage and their logs are the bulk of the volume).
            Pass None to fetch every job regardless of conclusion.
    """
    report = CIReport(run_id=run_id, repo=repo)
    job_ids = _list_job_ids(repo, run_id, conclusion)
    for job_id, job_name in job_ids.items():
        try:
            text = _fetch_job_log(repo, job_id)
        except subprocess.CalledProcessError:
            # Name first: the raw numeric id is the least useful identifier for exactly the leg
            # that failed ("tests (windows-latest, 3.8)" is what the reader needs).
            report.fetch_errors.append(f"{job_name} (job id {job_id})")
            continue
        analyze_log_text(text, job_id, job_name, report)
    return report


def main() -> None:
    """CLI entry point: fetch one run's job logs and print the consolidated report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo", help="owner/repo, e.g. fingoldo/mlframe")
    parser.add_argument("run_id", help="GitHub Actions run ID")
    parser.add_argument("--conclusion", default="failure", help="Only analyze jobs with this conclusion (default: failure). Use 'all' for every job.")
    parser.add_argument("--top", type=int, default=30, help="Number of top warning categories to print (default: 30)")
    args = parser.parse_args()

    conclusion = None if args.conclusion == "all" else args.conclusion
    report = analyze_run(args.repo, args.run_id, conclusion=conclusion)
    print(report.render(top=args.top))  # noqa: T201 -- this IS the CLI's output; a logger would send the report to stderr/nowhere


if __name__ == "__main__":
    main()

"""Pick the dependency versions a compatibility matrix should run, and write the README badge for the result.

A version matrix that lists every release is expensive and one that lists a few fixed versions goes stale. The legs are derived instead:

* the floor: the lowest version ``uv.lock`` resolves, i.e. the oldest version this repository is known to run on;
* the newest release of the floor's major series (``1.44.2`` for a ``1.36.1`` floor), where the series' last word is said;
* the newest release overall, labelled ``(latest)``.

Equal versions are run once (a series whose newest release is the floor, or a latest that is also the newest of the series), and pre-releases and
yanked files are ignored. The badge states the same list, e.g. ``1.36.1 | 1.44.2 | 2.0.1 (latest)``.

pyutilz has no lock file, so its floor is the lower bound the package declares for the extra being tested
(``--floor``, e.g. ``pandas>=1.5`` gives ``1.5``); a lock file, when present, is still accepted (``--lock``).

Usage::

    python scripts/ci_version_matrix.py plan --package pandas --floor 1.5 --python 3.12 --github-output "$GITHUB_OUTPUT"
    python scripts/ci_version_matrix.py plan --package polars --lock uv.lock --python 3.12 --github-output "$GITHUB_OUTPUT"
    python scripts/ci_version_matrix.py badge --package polars --legs '<matrix json>' --result success --out polars.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable, Optional

from packaging.version import InvalidVersion, Version

PYPI_URL = "https://pypi.org/pypi/{package}/json"
RETRIES = 5
BADGE_COLORS = {"success": "brightgreen", "failure": "red", "cancelled": "lightgrey", "skipped": "lightgrey"}


@dataclass(frozen=True)
class Leg:
    """One matrix leg: the exact version to install and the label shown in job names and on the badge."""

    version: str
    label: str


def lock_floor(lock_text: str, package: str) -> str:
    """The lowest version of ``package`` that ``uv.lock`` resolves (it can hold several, one per marker environment)."""
    pattern = re.compile(r'\[\[package\]\]\r?\nname = "' + re.escape(package) + r'"\r?\nversion = "([^"]+)"')
    versions = [Version(v) for v in pattern.findall(lock_text)]
    if not versions:
        raise LookupError(f"{package} is not in the lock file")
    return str(min(versions))


def usable_releases(releases: dict[str, list[dict]]) -> list[Version]:
    """Final releases that still have at least one non-yanked file, as sorted versions."""
    found: list[Version] = []
    for text, files in releases.items():
        try:
            version = Version(text)
        except InvalidVersion:
            continue
        if version.is_prerelease or version.is_devrelease or version.is_postrelease:
            continue
        if any(not f.get("yanked", False) for f in files):
            found.append(version)
    return sorted(found)


def plan(floor: str, releases: Iterable[Version]) -> list[Leg]:
    """The legs for ``floor`` and the available ``releases``: floor, newest of the floor's major series, newest overall; equal versions once."""
    available = sorted(set(releases))
    if not available:
        raise ValueError("no usable releases")
    floor_v = Version(floor)
    series = [v for v in available if v.major == floor_v.major and v >= floor_v]
    series_newest = max(series) if series else floor_v
    latest = max(available[-1], floor_v)
    legs: list[Leg] = []
    for version in (floor_v, series_newest, latest):
        # Compare as versions, not strings: "1.5" and "1.5.0" are the same release and must run once.
        if any(Version(leg.version) == version for leg in legs):
            continue
        legs.append(Leg(str(version), str(version)))
    legs = [Leg(leg.version, f"{leg.version} (latest)") if leg.version == str(latest) else leg for leg in legs]
    if len(legs) > 1 and legs[0].version == floor and legs[0].label == floor:
        legs[0] = Leg(legs[0].version, f"{floor} (min)")
    elif len(legs) == 1:
        legs[0] = Leg(legs[0].version, f"{legs[0].version} (min, latest)")
    return legs


def matrix(legs: Iterable[Leg], python_version: str) -> dict:
    """The ``strategy.matrix`` document for GitHub Actions: one include entry per leg."""
    return {"include": [{"version": leg.version, "label": leg.label, "python-version": python_version} for leg in legs]}


def badge(package: str, legs: Iterable[dict], result: str) -> dict:
    """The shields.io endpoint document: the tested versions as the message, coloured by the matrix result."""
    labels = [str(leg["label"]).replace(" (min)", "") for leg in legs]
    return {"schemaVersion": 1, "label": package, "message": " | ".join(labels) or "unknown", "color": BADGE_COLORS.get(result, "lightgrey")}


def fetch_releases(package: str, opener=urllib.request.urlopen, sleep=time.sleep) -> dict[str, list[dict]]:
    """The ``releases`` map of the package's PyPI JSON document, retrying on network errors."""
    last: Optional[Exception] = None
    for attempt in range(RETRIES):
        try:
            with opener(PYPI_URL.format(package=package), timeout=30) as response:  # nosec B310 - fixed https PyPI URL
                return json.loads(response.read().decode("utf-8"))["releases"]
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:  # noqa: PERF203 -- each retry attempt needs its own handler
            last = exc
            sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"PyPI lookup for {package} failed after {RETRIES} attempts: {last}")


def _plan_command(args: argparse.Namespace) -> int:
    """Print the matrix JSON for ``--package`` and, with ``--github-output``, append it as the ``matrix`` output."""
    if args.floor:
        floor = str(Version(args.floor))
    else:
        with open(args.lock, encoding="utf-8") as fh:
            floor = lock_floor(fh.read(), args.package)
    legs = plan(floor, usable_releases(fetch_releases(args.package)))
    document = json.dumps(matrix(legs, args.python), separators=(",", ":"))
    print(document)  # noqa: T201 -- the matrix JSON is this command's output, read from stdout by the workflow logs
    if args.github_output:
        with open(args.github_output, "a", encoding="utf-8") as out:
            out.write(f"matrix={document}\n")
    return 0


def _badge_command(args: argparse.Namespace) -> int:
    """Write the endpoint badge document to ``--out``."""
    legs = json.loads(args.legs)["include"]
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(badge(args.package, legs, args.result), fh)
        fh.write("\n")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_plan = sub.add_parser("plan")
    p_plan.add_argument("--package", required=True)
    source = p_plan.add_mutually_exclusive_group(required=True)
    source.add_argument("--floor", default="", help="the declared lower bound to test, e.g. 1.5")
    source.add_argument("--lock", default="", help="a lock file whose lowest resolved version is the floor")
    p_plan.add_argument("--python", default="3.12")
    p_plan.add_argument("--github-output", default="")
    p_plan.set_defaults(func=_plan_command)
    p_badge = sub.add_parser("badge")
    p_badge.add_argument("--package", required=True)
    p_badge.add_argument("--legs", required=True)
    p_badge.add_argument("--result", required=True)
    p_badge.add_argument("--out", required=True)
    p_badge.set_defaults(func=_badge_command)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

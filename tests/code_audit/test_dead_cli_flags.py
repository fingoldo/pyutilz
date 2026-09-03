"""Scanner tests for dead_cli_flags, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_dead_cli_flags,
)

from ._helpers import _write

# ---- dead_cli_flag ---------------------------------------------------------


def test_dead_cli_flag_never_read_flags(tmp_path: Path):
    _write(tmp_path, "bad.py", """
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()
    print(args.batch_size)
""")
    findings = scan_dead_cli_flags(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "dead_cli_flag"
    assert "resume" in findings[0].detail


def test_cli_flag_read_via_args_attr_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()
    if args.resume:
        print("resuming")
""")
    assert scan_dead_cli_flags(tmp_path) == []


def test_dead_cli_flag_version_action_not_flagged(tmp_path: Path):
    """--version, action="version" is a universal argparse idiom: the built-in action
    prints and exits internally, application code never reads args.version -- must
    never be flagged regardless of how common this exact shape is."""
    _write(tmp_path, "ok.py", """
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="version", version="%(prog)s 1.0")
    parser.parse_args()
""")
    assert scan_dead_cli_flags(tmp_path) == []


def test_dead_cli_flag_help_action_not_flagged(tmp_path: Path):
    """A manual add_argument("--info", action="help") is the same self-handling
    shape as the built-in -h/--help, just under a different flag name."""
    _write(tmp_path, "ok2.py", """
import argparse

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--info", action="help")
    parser.parse_args()
""")
    assert scan_dead_cli_flags(tmp_path) == []


def test_cli_flag_explicit_dest_used(tmp_path: Path):
    _write(tmp_path, "ok.py", """
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", dest="resume_flag", action="store_true")
    args = parser.parse_args()
    print(args.resume_flag)
""")
    assert scan_dead_cli_flags(tmp_path) == []


def test_cli_flag_used_in_other_file_of_tree_clean(tmp_path: Path):
    _write(tmp_path, "cli_def.py", """
import argparse

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    return parser
""")
    _write(tmp_path, "consumer.py", """
def run(args):
    if args.resume:
        pass
""")
    assert scan_dead_cli_flags(tmp_path) == []


def test_dead_cli_flag_ignores_selenium_options_add_argument(tmp_path: Path):
    """Regression (2026-07-22, false positive found in the wild in web/browser.py):
    Selenium's ChromeOptions/FirefoxOptions expose an UNRELATED add_argument(flag_string)
    method with the identical name -- it appends a raw command-line flag to a list passed to
    the external Chrome/Firefox binary, with no dest=/action=/etc. concept at all, so
    `.no_sandbox` is never expected to appear anywhere in this codebase's own Python source.
    Distinguished from real argparse usage by the absence of ANY keyword argument."""
    _write(tmp_path, "ok.py", """
from selenium.webdriver.chrome.options import Options

def start_selenium():
    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--remote-debugging-port=0")
    return options
""")
    assert scan_dead_cli_flags(tmp_path) == []


def test_dead_cli_flag_still_flags_argparse_flag_with_a_keyword(tmp_path: Path):
    """The zero-keyword-argument exclusion (added to stop flagging Selenium's unrelated
    add_argument) must not blind the scanner to a genuine dead argparse flag that carries at
    least one argparse-specific keyword -- the shape virtually all real argparse declarations
    use in practice (default=/action=/type=/help=/dest=)."""
    _write(tmp_path, "bad.py", """
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=False)
    args = parser.parse_args()
    print(args.batch_size)
""")
    findings = scan_dead_cli_flags(tmp_path)
    assert len(findings) == 1
    assert "resume" in findings[0].detail


def test_dead_cli_flag_known_limitation_zero_kwarg_argparse_flag_not_flagged(tmp_path: Path):
    """Documents an accepted trade-off: an argparse flag declared with NO keywords at all
    (bare `add_argument("--resume")`, relying entirely on argparse's defaults) is
    syntactically indistinguishable from Selenium's add_argument and is no longer flagged even
    if genuinely dead. Real argparse declarations in this codebase always carry at least one
    keyword (see dev/code_audit/cli.py), so this is a narrow, low-risk gap traded for
    eliminating a confirmed, concrete false-positive class."""
    _write(tmp_path, "bad_but_unflagged.py", """
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume")
    args = parser.parse_args()
    print(args.batch_size)
""")
    assert scan_dead_cli_flags(tmp_path) == []

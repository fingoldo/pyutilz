"""Scanner tests for mojibake, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_mojibake,
)

from ._helpers import _write

# ---- mojibake ---------------------------------------------------------------


def test_mojibake_roundtrip_corruption_flagged(tmp_path: Path):
    corrupted = "Русский".encode().decode("cp1251")
    _write(tmp_path, "bad.py", f"# {corrupted}\nx = 1\n")
    findings = scan_mojibake(tmp_path)
    assert len(findings) == 1, findings
    assert findings[0].check == "mojibake"


def test_mojibake_genuine_cyrillic_not_flagged(tmp_path: Path):
    _write(tmp_path, "ok.py", "# Привет мир\nx = 1\n")
    assert scan_mojibake(tmp_path) == []


def test_mojibake_ascii_only_not_flagged(tmp_path: Path):
    _write(tmp_path, "ok.py", "# a normal comment\nx = 1\n")
    assert scan_mojibake(tmp_path) == []


def test_mojibake_short_cyrillic_regex_range_not_flagged(tmp_path: Path):
    # Real false positive found while dogfooding this scanner on pyutilz itself
    # (src/pyutilz/text/humanizer.py): a regex character class like [A-ZА-ЯЁ] gets split
    # by the ASCII "-" into a short 2-char Cyrillic run ("ЯЁ") that coincidentally
    # round-trips through cp1251-encode -> utf-8-decode into different, legible-looking
    # text -- purely by chance, not because anything is actually corrupted.
    _write(tmp_path, "ok.py", 'hits = [m.start() for m in re.finditer(r"\\. [A-ZА-ЯЁ]", text)]\n')
    assert scan_mojibake(tmp_path) == []


# ---- F214: an undecodable file is the strongest mojibake signal ----------------------


def test_mojibake_reports_a_file_that_is_not_utf8(tmp_path: Path):
    (tmp_path / "bad.py").write_bytes(b"x = 1  # \xef\xf0\xe8\xe2\xe5\xf2\n")
    findings = scan_mojibake(tmp_path)
    assert len(findings) == 1 and "not valid UTF-8" in findings[0].detail


def test_mojibake_accepts_a_clean_utf8_file(tmp_path: Path):
    _write(tmp_path, "ok.py", "x = 1\n")
    assert scan_mojibake(tmp_path) == []

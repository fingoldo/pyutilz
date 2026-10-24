import io
import re
import tokenize
import subprocess  # nosec B404 - runs py_ci_shared's pin check with a fixed argv, no shell
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


SRC_ROOT = REPO_ROOT / "src" / "pyutilz"


def test_f01_no_batch_stamped_type_ignore_justification():
    """No `# type: ignore` in src carries the batch-applied "json/external lib/dynamic attr" reason.

    That single string was pasted verbatim onto 54 sites and was demonstrably false at several of
    them (numpy `.nbytes` arithmetic, a locally-built set). Every remaining ignore names its own cause.
    """
    stamped = [
        f"{path.relative_to(SRC_ROOT).as_posix()}:{tok.start[0]}"
        for path in SRC_ROOT.rglob("*.py")
        for tok in tokenize.generate_tokens(io.StringIO(path.read_text(encoding="utf-8")).readline)
        if tok.type == tokenize.COMMENT and "untyped upstream source (json/external lib/dynamic attr)" in tok.string
    ]
    assert stamped == [], f"batch-stamped type-ignore justification is back at: {stamped}"  # audit-exempt: source_text_assertion -- regression guard scanning src for a banned justification string; reading source is its job


@pytest.mark.skipif(sys.version_info < (3, 9), reason="py-ci-shared requires python>=3.9")
def test_f06_pin_check_reports_a_mismatch():
    """`language: system` means the hook runs whatever ruff is installed; this gate is what says so."""
    import py_ci_shared.pinned_tool_versions  # a missing install fails; only the 3.8 legs skip, via the mark  # noqa: F401
    out = subprocess.run(  # nosec B603 - fixed argv (sys.executable plus literal flags), shell=False
        [sys.executable, "-m", "py_ci_shared.pinned_tool_versions"], capture_output=True, text=True, check=False, cwd=str(REPO_ROOT)
    )
    # Either the box matches the pin (exit 0) or it does not and the mismatch is NAMED; anything else, a crash
    # included, fails.
    assert out.returncode == 0 or "pinned-tool-version mismatch" in out.stdout, (out.returncode, out.stdout, out.stderr)


@pytest.mark.skipif(sys.version_info < (3, 9), reason="py-ci-shared requires python>=3.9")
def test_f06_pin_check_fails_on_a_different_pin(tmp_path):
    """The gate has teeth: a pyproject pinning another ruff than the shared version fails it."""
    import py_ci_shared.pinned_tool_versions  # a missing install fails; only the 3.8 legs skip, via the mark  # noqa: F401
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    bad = re.sub(r'"ruff==[0-9][^";\s]*', '"ruff==0.0.1', text, count=1)
    assert bad != text, "pyproject.toml has no exact ruff pin to alter"
    (tmp_path / "pyproject.toml").write_text(bad, encoding="utf-8")
    out = subprocess.run(  # nosec B603 - fixed argv (sys.executable plus a tmp path), shell=False
        [sys.executable, "-m", "py_ci_shared.pinned_tool_versions", "--pyproject", str(tmp_path / "pyproject.toml")],
        capture_output=True, text=True, check=False, cwd=str(REPO_ROOT),
    )
    assert out.returncode == 1 and "0.0.1" in out.stdout, out.stdout


def test_f06_pin_check_is_wired_as_a_hook():
    text = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "py_ci_shared.pinned_tool_versions" in text


def test_f09_benchmarks_is_no_longer_excluded_from_ruff():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    precommit = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert 'exclude = [".git", "__pycache__", "build", "dist"]' in pyproject
    assert '"_benchmarks/**"' in pyproject
    assert "exclude: (^|/)_benchmarks/" not in precommit


def test_f10_platform_ignores_name_their_environment():
    """The Windows-only / numba ignores stay (closed decision) but must say WHICH environment."""
    bare = []
    seen = 0
    for rel in ("system/system/memory.py", "system/system/misc.py", "text/similarity/_numba_kernels.py"):
        for line in (SRC_ROOT / rel).read_text(encoding="utf-8").splitlines():
            if "# type: ignore[attr-defined]" in line:
                seen += 1
                if not line.split("# type: ignore[attr-defined]", 1)[1].strip():
                    bare.append(f"{rel}: {line.strip()}")
    # Assert the precondition too: if the ignores were all deleted (the stronger failure, since the
    # closed decision says they must stay), an all-conditional check would pass by seeing nothing.
    assert seen == 8, f"expected the 8 platform-conditional attr-defined ignores, found {seen}"
    assert bare == [], f"attr-defined ignore(s) with no environment note: {bare}"


_INVISIBLE = tuple(chr(cp) for cp in (0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF))


def test_f11_no_invisible_characters_in_python_sources():
    """A U+200B inside a docstring is undiagnosable by reading the file, and docstrings are load-bearing here."""
    offenders = []
    for base in ("src", "tests", "scripts", "_benchmarks"):
        for path in (REPO_ROOT / base).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for ch in _INVISIBLE:
                if ch in text:
                    offenders.append(f"{path.relative_to(REPO_ROOT).as_posix()}: {ch!r}")
    assert offenders == [], f"invisible character(s) in source: {offenders}"


def test_f11_invisible_character_rules_are_selected():
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for code in ("PLE2510", "PLE2512", "PLE2513", "PLE2514", "PLE2515", "RET503"):
        assert f'"{code}"' in text

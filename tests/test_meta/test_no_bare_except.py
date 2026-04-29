"""H3 — meta-test that no production code uses ``except:`` (bare) or
``except BaseException:``.

Bare ``except:`` swallows EVERYTHING — ``KeyboardInterrupt`` (so the
user can't Ctrl-C), ``SystemExit`` (forks misbehave), and
``MemoryError`` (debugger gets confused). It also masks bugs in the
try-block by catching them as if they were expected. The narrow form
``except Exception:`` is the safe equivalent in nearly every case.

Catches:
  - ``except:``                     → must be ``except Exception:`` (or specific)
  - ``except BaseException:``       → same (or KeyboardInterrupt / SystemExit individually if intended)

Skips ``except Exception:`` and any narrower exception type — those
are intentional.

Snapshot-style; first run captures any existing offenders. Future
commits adding new bare excepts fail.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

import pyutilz

PYUTILZ_DIR = Path(pyutilz.__file__).resolve().parent
_BASELINE_PATH = Path(__file__).resolve().parent / "_bare_except_baseline.json"

_EXEMPT_PATH_FRAGMENTS = ("__pycache__", "tests")


def _refresh_requested() -> bool:
    return "--refresh-bare-except-baseline" in sys.argv


def _is_bare_except(handler: ast.ExceptHandler) -> bool:
    """``handler.type`` is None for bare ``except:``; or a Name
    ``BaseException`` for the equivalent dangerous form.

    EXCEPTION: ``except BaseException as e: ... raise`` (re-raise) is
    legitimate — phase-tracking context managers, request-scope cleanup,
    and similar patterns that audit EVERY exit path. We detect a bare
    re-raise inside the handler and allow that case.
    """
    if handler.type is None:
        return True
    if isinstance(handler.type, ast.Name) and handler.type.id == "BaseException":
        for sub in ast.walk(handler):
            if isinstance(sub, ast.Raise) and sub.exc is None:
                return False
        return True
    return False


def _build_offending_set() -> set[str]:
    out: set[str] = set()
    for py in PYUTILZ_DIR.rglob("*.py"):
        if any(frag in py.parts for frag in _EXEMPT_PATH_FRAGMENTS):
            continue
        if py.name.endswith(".py.old"):
            continue
        try:
            src = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        rel = py.relative_to(PYUTILZ_DIR).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            for handler in node.handlers:
                if _is_bare_except(handler):
                    out.add(f"{rel}:{handler.lineno}")
    return out


def test_no_new_bare_except_clauses():
    current = _build_offending_set()

    if _refresh_requested() or not _BASELINE_PATH.exists():
        _BASELINE_PATH.write_text(
            json.dumps(sorted(current), indent=2), encoding="utf-8"
        )
        pytest.skip(
            f"bare-except baseline refreshed at {_BASELINE_PATH.name} "
            f"({len(current)} bare clauses)"
        )

    baseline = set(json.loads(_BASELINE_PATH.read_text(encoding="utf-8")))
    new = sorted(current - baseline)
    fixed = sorted(baseline - current)

    if fixed:
        sys.stderr.write(
            f"\n[test_no_new_bare_except_clauses] {len(fixed)} site(s) "
            f"DRAINED:\n  " + "\n  ".join(fixed[:15])
            + (f"\n  ... and {len(fixed) - 15} more" if len(fixed) > 15 else "")
            + "\n  Refresh: pytest ... --refresh-bare-except-baseline\n"
        )

    if new:
        pytest.fail(
            f"{len(new)} new bare ``except:`` (or ``except BaseException:``) "
            f"clause(s). Replace with ``except Exception:`` or a narrower "
            f"specific exception type — bare-except swallows "
            f"KeyboardInterrupt/SystemExit and masks real bugs:\n  "
            + "\n  ".join(new[:30])
            + (f"\n  ... and {len(new) - 30} more" if len(new) > 30 else "")
        )

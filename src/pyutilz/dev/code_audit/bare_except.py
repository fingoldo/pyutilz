"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _line_text, _read_src_lines, _safe_parse

# --- bare `except:` / `except BaseException:` ----------------------------


def _is_bare_except(handler: ast.ExceptHandler) -> bool:
    """``handler.type`` is None for bare ``except:``; or a Name ``BaseException`` for the
    equivalent dangerous form.

    EXCEPTION: ``except BaseException as e: ... raise`` (bare re-raise) is legitimate --
    phase-tracking context managers, request-scope cleanup, and similar patterns that audit EVERY
    exit path. A bare re-raise inside the handler is allowed.
    """
    if handler.type is not None and not (isinstance(handler.type, ast.Name) and handler.type.id == "BaseException"):
        return False
    # Bare re-raise exemption applies to BOTH shapes -- a bare `except:` whose body immediately
    # re-raises swallows nothing either, exactly as the docstring promises.
    for sub in ast.walk(handler):
        if isinstance(sub, ast.Raise) and sub.exc is None:
            return False
    return True


def scan_bare_except(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find ``except:`` (bare) and ``except BaseException:`` clauses.

    Bare ``except:`` swallows EVERYTHING -- ``KeyboardInterrupt`` (the user can't Ctrl-C),
    ``SystemExit`` (forks misbehave), and ``MemoryError`` (debugger gets confused). It also masks
    bugs in the try-block by catching them as if they were expected. ``except Exception:`` is the
    safe equivalent in nearly every case. A handler that immediately re-raises (bare ``raise``) is
    not flagged -- auditing every exit path via ``except BaseException: ... raise`` is legitimate.

    Distinct from ``scan_broad_except_swallows``, which targets ``except Exception:`` handlers
    that swallow silently -- this scanner targets the strictly WIDER (and never intentional)
    ``except:``/``except BaseException:`` shape.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = _read_src_lines(py)
        rel = py.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            for handler in node.handlers:
                if not _is_bare_except(handler):
                    continue
                findings.append(Finding(
                    check="bare_except",
                    severity="P1",
                    file=rel,
                    line=handler.lineno,
                    snippet=_line_text(src_lines, handler.lineno),
                    detail=(
                        "bare `except:` (or `except BaseException:`) swallows KeyboardInterrupt/"
                        "SystemExit and masks real bugs. Replace with `except Exception:` or a "
                        "narrower specific exception type."
                    ),
                ))
    return findings

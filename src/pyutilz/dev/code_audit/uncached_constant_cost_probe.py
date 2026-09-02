"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text

# --- uncached constant-cost probe ----------------------------------------

# Decorators that already memoize the whole call.
_CACHING_DECORATORS = ("lru_cache", "cache", "cached_property", "cached", "memoize", "once")

# (dotted-call fragment, what it costs) -- the probe families whose cost is effectively constant
# per process, so paying it on every call is pure waste.
_PROBES: tuple[tuple[str, str], ...] = (
    ("subprocess.run", "spawns a process"),
    ("subprocess.check_output", "spawns a process"),
    ("subprocess.check_call", "spawns a process"),
    ("subprocess.call", "spawns a process"),
    ("subprocess.Popen", "spawns a process"),
    ("ctypes.WinDLL", "loads a shared library"),
    ("ctypes.CDLL", "loads a shared library"),
    ("ctypes.windll.LoadLibrary", "loads a shared library"),
    ("os.makedirs", "hits the filesystem (measurably slower than exists-then-skip)"),
    ("Path.mkdir", "hits the filesystem"),
    ("importlib.import_module", "walks the import system"),
    ("importlib.util.find_spec", "walks the import system"),
    ("shutil.which", "scans every PATH entry"),
    ("platform.processor", "shells out on some platforms"),
    ("socket.gethostbyname", "performs a DNS lookup"),
)


def _dotted_name(node: ast.AST) -> str:
    """Render ``a.b.c`` / ``c`` from a Call's func node, or ``""``."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    elif parts:
        parts.append("")
    return ".".join(reversed(parts))


def _is_cached(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True if ``func`` carries any decorator that memoizes its result, so repeated calls cost nothing."""
    text = " ".join(ast.dump(d) for d in func.decorator_list)
    return any(name in text for name in _CACHING_DECORATORS)


def _has_module_level_memo(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether the body writes a module-level memo (``global _cached_x`` / ``nonlocal``), which is
    the hand-rolled equivalent of a caching decorator."""
    return any(isinstance(node, (ast.Global, ast.Nonlocal)) for node in ast.walk(func))


def _takes_only_config_parameters(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True when every parameter has a default (or there are none), ignoring ``self``/``cls``.

    A function whose result varies with a REQUIRED argument is not constant-cost by construction,
    so restricting to no-argument / all-defaulted functions is what keeps the false-positive rate
    survivable: those are the ones whose answer is the same every time it is asked.
    """
    a = func.args
    positional = [*a.posonlyargs, *a.args]
    if positional and positional[0].arg in ("self", "cls"):
        positional = positional[1:]
    if a.vararg or a.kwarg:
        return False
    n_defaultable = len(positional) + len(a.kwonlyargs)
    n_defaults = len([d for d in a.defaults if d is not None]) + len([d for d in a.kw_defaults if d is not None])
    return n_defaults >= n_defaultable


def _matching_probe(dotted: str) -> "tuple[str, str] | None":
    """Match a dotted call name against ``_PROBES`` on its trailing segments.

    Trailing-segment matching (not full equality) so an aliased import -- ``sp.run``, ``from
    subprocess import run`` -- is recognised, while a same-named method on an unrelated object
    (``self.run``, ``pool.map``) is not: the last segment alone is never enough, the probe's own
    module segment must be present or the call must be a bare use of the probe's function name.
    """
    segments = dotted.split(".")
    for probe, cost in _PROBES:
        probe_segments = probe.split(".")
        if segments[-len(probe_segments) :] == probe_segments:
            return probe, cost
        if len(segments) == 1 and segments[0] == probe_segments[-1]:
            return probe, cost
    return None


def scan_uncached_constant_cost_probe(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
) -> list[Finding]:
    """Find expensive, effectively-constant probes repeated on every call.

    The shape: a helper shells out to ``nvidia-smi``, rebuilds a ``ctypes.WinDLL`` handle, or calls
    ``os.makedirs(..., exist_ok=True)`` on a directory that already exists - and does so per call,
    from a dispatch path taken thousands of times. The answer does not change within a process, so
    every repetition after the first is pure overhead on a hot path.

    Reported when all of the following hold:

    1. the function takes no parameters, or only defaulted ones (so its answer cannot vary with a
       required input);
    2. its body calls one of the probe families above;
    3. it carries no caching decorator and writes no module-level memo.

    Severity: P2, warn-only, and opt-in rather than part of the default sweep. A probe that MUST be
    fresh - a liveness check, a config reload - is indistinguishable from one that must not, and
    the honest example is ``_pid_alive``: the pid check genuinely has to re-run, while the
    ``WinDLL`` handle it rebuilds each time does not. So this names a function to look at and says
    which call inside it is the constant-cost part; the judgement stays human.
    """
    findings: list[Finding] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = py.relative_to(root).as_posix()

        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if _is_cached(func) or _has_module_level_memo(func) or not _takes_only_config_parameters(func):
                continue

            for node in ast.walk(func):
                if not isinstance(node, ast.Call):
                    continue
                dotted = _dotted_name(node.func)
                if not dotted:
                    continue
                match = _matching_probe(dotted)
                if match is None:
                    continue
                _probe, cost = match
                findings.append(
                    Finding(
                        check="uncached_constant_cost_probe",
                        severity="P2",
                        file=rel,
                        line=node.lineno,
                        snippet=_line_text(src_lines, node.lineno),
                        detail=(
                            f"`{func.name}()` takes no varying input yet calls `{dotted}` on every call, "
                            f"which {cost}. Nothing memoizes the result (no lru_cache/cache/cached_property, "
                            f"no module-level memo), so a hot dispatch path pays this cost per decision. "
                            f"Cache it, unless this probe genuinely has to be re-taken every time."
                        ),
                    )
                )
                break  # one finding per function is enough to route it into triage
    return findings

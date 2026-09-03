"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""

from __future__ import annotations

import ast
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _own_nodes, _line_text, _read_src_lines, _safe_parse

# --- uncached constant-cost probe ----------------------------------------

# Decorators that already memoize the whole call. Matched STRUCTURALLY on the decorator's own
# name -- a substring search over `ast.dump` let any decorator whose ARGUMENTS merely contain the
# text (`@app.route("/cache")`) exempt the function. Under structural matching `lru_cache` is no
# longer subsumed by `cache`, so both entries carry their weight.
_CACHING_DECORATORS = frozenset({"lru_cache", "cache", "cached_property", "cached", "memoize", "once"})

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


def _decorator_name(node: ast.expr) -> str:
    """The bare name of a decorator expression: ``@lru_cache``, ``@functools.lru_cache(maxsize=1)`` -> ``lru_cache``."""
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _is_cached(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True if ``func`` carries any decorator that memoizes its result, so repeated calls cost nothing."""
    return any(_decorator_name(d) in _CACHING_DECORATORS for d in func.decorator_list)


def _has_module_level_memo(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether the body WRITES a module-level memo (``global _cached_x`` and then assigns it).

    The declaration alone is not a memo: an unrelated ``global counter`` used only for a statistic
    exempted a genuinely uncached probe. The declared name must actually be stored to.
    """
    declared: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            declared.update(node.names)
    if not declared:
        return False
    for node in ast.walk(func):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in declared:
            return True
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store) and isinstance(node.value, ast.Name) and node.value.id in declared:
            return True
    return False


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


def _probe_imports(tree: ast.Module) -> dict[str, str]:
    """Local name -> its probe module, for `from subprocess import run` style imports.

    A single-segment call name is only a probe when the file actually imported it from the probe's
    module; without this a locally defined `def run()` was reported as "spawns a process" in a file
    with no imports at all.
    """
    probe_modules = {probe.split(".")[0] for probe, _cost in _PROBES}
    bound: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in probe_modules and node.level == 0:
            for alias in node.names:
                bound[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return bound


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
        src_lines = _read_src_lines(py)
        rel = py.relative_to(root).as_posix()

        probe_imports = _probe_imports(tree)

        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if _is_cached(func) or _has_module_level_memo(func) or not _takes_only_config_parameters(func):
                continue

            for node in _own_nodes(func):
                if not isinstance(node, ast.Call):
                    continue
                dotted = _dotted_name(node.func)
                if not dotted:
                    continue
                match = _matching_probe(dotted)
                if match is None and isinstance(node.func, ast.Name) and node.func.id in probe_imports:
                    match = _matching_probe(probe_imports[node.func.id])
                if match is None and isinstance(node.func, ast.Attribute) and node.func.attr == "mkdir":
                    # `p.mkdir(...)` / `Path(x).mkdir(...)` -- the spelling that is actually written.
                    # `_dotted_name` renders the receiver as "" for anything but a Name chain, so the
                    # registered `Path.mkdir` entry could only ever match `Path.mkdir(p)`, which nobody writes.
                    receiver = node.func.value
                    receiver_name = receiver.func.id if isinstance(receiver, ast.Call) and isinstance(receiver.func, ast.Name) else (receiver.id if isinstance(receiver, ast.Name) else None)
                    if receiver_name in (None, "Path") or receiver_name not in {"os", "shutil", "subprocess"}:
                        match = ("Path.mkdir", "hits the filesystem")
                        dotted = f"{receiver_name}.mkdir" if receiver_name else "Path(...).mkdir"
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

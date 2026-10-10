"""Core on-disk kernel-tuning cache: hardware fingerprinting, cache-dir resolution, and provenance tracking."""
from __future__ import annotations

import contextlib
import datetime as _dt
import errno
import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional

from pyutilz.system.gpu_dispatch import gpu_capability_summary

logger = logging.getLogger(__name__)

# v4: SAME file layout as v3 -- bumped purely to INVALIDATE every previously-recorded tuning. Until
# 2026-09-02 ``benchmark.time_backend`` timed backends with NO device synchronize, so every GPU
# ``wall_ms`` it persisted was launch overhead rather than compute (measured 0.0366 ms vs 69.42 ms
# synchronized on a cupy 4000x4000 matmul -- a 1894x under-count). Those regions picked a GPU
# variant unconditionally and, being immutable + cached per host, would never be re-measured. A
# schema bump is the cache's existing invalidation seam, so v3 files are now skipped on read and
# re-tuned with honest (synchronized) timings.
# v3: immutable per-(host,kernel,code_version) files + O_EXCL sweep markers (was v2: monolithic JSON).
SCHEMA_VERSION = 4

# How long an INPROGRESS sweep marker is trusted before a would-be sweeper may
# STEAL it (owner crashed / hung). The expensive GPU/CPU sweeps run hundreds of
# seconds; this budget must exceed a legitimate sweep but bound the wedge from a
# killed-mid-sweep process. Override via $PYUTILZ_KERNEL_SWEEP_BUDGET_SEC.
_DEFAULT_SWEEP_BUDGET_SECONDS = 1800.0

# Sentinel used in a filename when a tuning carries no code_version (update()
# may be called without one). A real code_version is a SHA-256 hex string.
_NO_CODE_VERSION = "nocv"

def _slug(s: str, maxlen: int = 40) -> str:
    """Filename-safe lowercase + truncated form of an arbitrary string."""
    s = re.sub(r"\(R\)|\(TM\)|\bCPU\b|\bGPU\b|@.*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", "-", s.strip())
    s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
    return s.strip("-._").lower()[:maxlen] or "unknown"


@lru_cache(maxsize=1)
def _cpu_model_slug() -> str:
    """Filename-safe slug of the CPU's brand string (e.g. "unknown" on probe failure), cached for the process lifetime."""
    # cpuinfo.get_cpu_info() runs a ~2s WMI/CPUID probe on Windows (and can
    # stall under load); the CPU model is invariant, so cache it for the process
    # lifetime. This lru survives hw_fingerprint.cache_clear() (which tests call
    # every test), so cpuinfo runs at most ONCE per process regardless of how
    # often the fingerprint lru or the cache-dir is reset.
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        return _slug(info.get("brand_raw", "unknown"))
    except Exception as e:
        logger.debug("cpuinfo probe failed (%s), CPU slug falls back to 'unknown'", e)
        return "unknown"


def _current_device_id() -> int:
    """Return the live CUDA device id (whatever the caller is using).
    Falls back to 0 on probe failure. Lets the cache key reflect e.g.
    ``device=1`` on a 2-GPU box where the user routed mlframe to a
    non-default device."""
    try:
        import cupy as cp
        return int(cp.cuda.runtime.getDevice())
    except Exception as e:
        logger.debug("Current CUDA device probe failed (%s), falling back to device 0", e)
        return 0


@lru_cache(maxsize=16)
def _gpu_summary_cached(device_id: int):
    """Per-device GPU capability probe (nvidia-smi / gputil / cupy query --
    ~0.1-2s and can stall under load), cached for the process. Keyed BY device
    id so a multi-GPU box gets a distinct cached summary per device (the
    per-device fingerprint + the multi-GPU sweep rely on this). CPU is cached
    globally (_cpu_model_slug -- one CPU); GPUs must be per-device. Tests that
    mock gpu_capability_summary must call _gpu_summary_cached.cache_clear()."""
    # Resolve ``gpu_capability_summary`` through the FACADE package so a
    # ``mock.patch.object(cache, "gpu_capability_summary", ...)`` on the public
    # package (as the HW-fingerprint tests do) is honored; falls back to the real
    # import when unpatched.
    import sys as _sys
    _facade = _sys.modules.get("pyutilz.performance.kernel_tuning.cache")
    _probe = getattr(_facade, "gpu_capability_summary", gpu_capability_summary)
    return _probe(device_id)


_GPU_NONE = "no-gpu"
_GPU_UNKNOWN = "gpu-unknown"
# CUDA runtime statuses meaning "no usable device / driver on this host" (cudaErrorInsufficientDriver, cudaErrorNoDevice): a definite answer, not a probe failure.
_CUDA_NO_DEVICE_STATUSES = (35, 100)
_CUPY_DIST_NAMES = ("cupy", "cupy-cuda11x", "cupy-cuda12x", "cupy-cuda13x")
_GPU_OPT_OUT_PREDICATES: List[Callable[[], bool]] = []


def register_gpu_opt_out(predicate: Callable[[], bool]) -> None:
    """Register an extra "this run must not use the GPU" predicate (idempotent); an opted-out run neither reads nor writes the persisted fingerprint."""
    if predicate not in _GPU_OPT_OUT_PREDICATES:
        _GPU_OPT_OUT_PREDICATES.append(predicate)


def _predicate_holds(predicate: Callable[[], bool]) -> bool:
    """Result of an opt-out predicate; a predicate that raises counts as not opting out."""
    try:
        return bool(predicate())
    except Exception as e:
        logger.warning("GPU opt-out predicate failed (%s), treating it as not opting out", e)
        return False


def _gpu_opted_out() -> bool:
    """True when this run declared it must not use the GPU: ``CUDA_VISIBLE_DEVICES`` empty / ``-1`` / ``NoDevFiles``, ``PYUTILZ_DISABLE_GPU=1``,
    or any predicate passed to :func:`register_gpu_opt_out`."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() in ("", "-1", "NoDevFiles"):
        return True
    if os.environ.get("PYUTILZ_DISABLE_GPU", "").strip() == "1":
        return True
    return any(_predicate_holds(predicate) for predicate in _GPU_OPT_OUT_PREDICATES)


def _gpu_device_count() -> int:
    """Number of CUDA devices. ``0`` only when the stack reports it without error (no driver / no device); any other failure propagates."""
    try:
        import cupy as cp
    except ImportError:
        cp = None
    if cp is not None:
        try:
            return int(cp.cuda.runtime.getDeviceCount())
        except cp.cuda.runtime.CUDARuntimeError as e:
            if getattr(e, "status", None) in _CUDA_NO_DEVICE_STATUSES:
                return 0
            raise
    try:
        from numba import cuda as numba_cuda
    except ImportError:
        return 0
    try:
        return len(numba_cuda.list_devices())
    except numba_cuda.CudaSupportError as e:
        logger.warning("numba reports no usable CUDA driver (%s), treating the host as GPU-less", e)
        return 0


def _gpu_slug_and_cc() -> tuple[str, str]:
    """Returns ``(gpu_name_slug, cc_str)``.

    * ``("no-gpu", "")`` -- the host genuinely has no CUDA device (the stack reported zero devices without error).
    * ``("gpu-unknown", "")`` -- the probe failed, or the stack reports devices the capability probe could not describe. NOT a statement about the
      hardware: :func:`hw_fingerprint` never persists it.

    Uses the LIVE current CUDA device id, not always 0, so a 2-GPU box where the user routes to device 1 gets a distinct fingerprint.
    """
    try:
        dev_id = _current_device_id()
        summary = _gpu_summary_cached(dev_id)
        if summary is None:
            return (_GPU_NONE, "") if _gpu_device_count() == 0 else (_GPU_UNKNOWN, "")
        name = summary.get("name") or "unknown"
        cc = f"{int(summary.get('cc_major', 0))}.{int(summary.get('cc_minor', 0))}"
        return (_slug(name), cc)
    except Exception as e:
        logger.debug("gpu_capability_summary failed: %s", e)
        return (_GPU_UNKNOWN, "")


def _token(value: object) -> str:
    """Filename-safe fragment of ``value`` (``x`` when it is missing)."""
    text = re.sub(r"[^A-Za-z0-9.]+", "", "" if value is None else str(value))
    return text if text else "x"


def _one_dist_version(name: str) -> Optional[str]:
    """Installed version of distribution ``name`` as a key token, or ``None`` when it is not installed."""
    try:
        from importlib import metadata

        return _token(metadata.version(name))
    except Exception as e:
        logger.debug("distribution %s version unavailable (%s)", name, e)
        return None


def _dist_version(*names: str) -> str:
    """Installed version of the first distribution in ``names`` that exists, without importing it; ``x`` when none is installed."""
    versions = (_one_dist_version(name) for name in names)
    return next((v for v in versions if v is not None), "x")


def _package_stamp() -> str:
    """Versions of numba and cupy as installed, cheap enough to re-check on every disk read."""
    return f"nb{_dist_version('numba')}_cupy{_dist_version(*_CUPY_DIST_NAMES)}"


def _numba_threads() -> int:
    """Effective numba thread count of this process (``NUMBA_NUM_THREADS``), falling back to the CPU count when numba is absent."""
    try:
        import numba

        return int(getattr(numba.config, "NUMBA_NUM_THREADS", os.cpu_count() or 1))
    except Exception as e:
        logger.debug("numba thread count unavailable (%s), using the CPU count", e)
        return int(os.cpu_count() or 1)


def _vram_class(total_vram: object) -> str:
    """Total VRAM rounded up to a power-of-two GiB class (``vram8g``); GPUtil reports MiB, the summary key says GB, so both scales are accepted."""
    try:
        value = float(total_vram)  # type: ignore[arg-type]  # object from a loosely-typed summary dict; the except covers a non-numeric value
    except (TypeError, ValueError):
        return "vramx"
    if value <= 0:
        return "vramx"
    gib = value / 1024.0 if value > 256 else value
    cls = 1
    while cls < gib:
        cls *= 2
    return f"vram{cls}g"


def _gpu_identity_suffix() -> str:
    """Device index, VRAM class, CUDA driver / runtime and cupy versions of the live GPU, as a key fragment. Fields that cannot be read are ``x``."""
    prov = _build_provenance()
    gpu = prov.get("gpu_summary")
    vram = gpu.get("total_vram_gb") if isinstance(gpu, dict) else None
    return "_".join(
        (
            f"d{_current_device_id()}",
            _vram_class(vram),
            f"drv{_token(prov.get('cuda_driver_version'))}",
            f"rt{_token(prov.get('cuda_runtime_version'))}",
        )
    )


_HW_FP_DISK_FILENAME = ".hw_fingerprint.json"
_HW_FP_SCHEMA_VERSION = 2
_HW_FP_FRESHNESS_SECONDS = 7 * 24 * 3600  # 7 days


def _hw_fp_selector() -> str:
    """Which devices this process can see: persisted entries are keyed by it so a run routed to another GPU does not reuse another device's key."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    return visible if visible else "default"


def _hw_fp_path() -> Optional[str]:
    """Path of the on-disk fingerprint file, or ``None`` when the cache directory cannot be resolved."""
    try:
        return os.path.join(cache_dir(), _HW_FP_DISK_FILENAME)
    except Exception as e:
        logger.debug("Could not resolve on-disk hw-fingerprint path (%s), skipping disk cache", e)
        return None


def _load_hw_fp_entries(path: str) -> Dict[str, Any]:
    """Entries of a schema-compatible fingerprint file (``{}`` when it is missing, unreadable or of another schema)."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("schema_version") != _HW_FP_SCHEMA_VERSION:
        return {}
    entries = data.get("entries")
    return entries if isinstance(entries, dict) else {}


def _read_hw_fingerprint_from_disk(selector: Optional[str] = None) -> Optional[str]:
    """Return the persisted hardware part of the fingerprint for ``selector`` (default: the current ``CUDA_VISIBLE_DEVICES`` view) when it is
    schema-compatible, recent enough and was recorded under the installed numba / cupy versions; ``None`` otherwise.

    ``PYUTILZ_HW_FP_REFRESH=1`` forces a recompute even on a fresh entry (a driver or GPU swap does not change the file by itself).
    """
    if os.environ.get("PYUTILZ_HW_FP_REFRESH", "").strip() == "1":
        return None
    path = _hw_fp_path()
    if path is None:
        return None
    entry = _load_hw_fp_entries(path).get(selector if selector is not None else _hw_fp_selector())
    if not isinstance(entry, dict):
        return None
    fp, ts = entry.get("fingerprint"), entry.get("ts")
    if not isinstance(fp, str) or not fp or not isinstance(ts, (int, float)):
        return None
    if max(0.0, time.time() - float(ts)) > _HW_FP_FRESHNESS_SECONDS:
        return None
    if entry.get("stamp") != _package_stamp():
        return None
    return fp


def _write_hw_fingerprint_to_disk(fingerprint: str, selector: Optional[str] = None) -> None:
    """Persist the freshly-computed hardware fingerprint under ``selector``. Best-effort: silently swallows write errors (read-only homedir,
    permissions, etc.) so the in-memory lru_cache still works. Callers must not pass a fingerprint derived from a failed or opted-out probe."""
    tmp: Optional[str] = None
    try:
        path = _hw_fp_path()
        if path is None:
            return
        entries = _load_hw_fp_entries(path)
        entries[selector if selector is not None else _hw_fp_selector()] = {
            "fingerprint": fingerprint,
            "ts": time.time(),
            "stamp": _package_stamp(),
            "ts_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        }
        # Unique temp name per writer: per-target training scripts start concurrently, and a shared ``path + ".tmp"``
        # let one process ``os.replace`` a file the other was still writing.
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"schema_version": _HW_FP_SCHEMA_VERSION, "entries": entries}, f)
        os.replace(tmp, path)
    except Exception as e:
        logger.debug("hw_fingerprint: failed to persist to disk: %s", e)
        if tmp is not None:
            with contextlib.suppress(OSError):
                os.remove(tmp)


@lru_cache(maxsize=1)
def hw_fingerprint() -> str:
    """Per-host key. Format::

        cpu_<cpu>_gpu_<gpu>_cc<M.m>_d<device>_vram<N>g_drv<driver>_rt<runtime>_nb<numba>_cupy<cupy>_t<threads>
        cpu_<cpu>_no-gpu_nb<numba>_cupy<cupy>_t<threads>      (host with no CUDA device)
        cpu_<cpu>_gpu-unknown_nb<numba>_cupy<cupy>_t<threads>  (GPU probe failed; never persisted)

    ``threads`` is the effective numba thread count of THIS process, so a tuning measured with 2 threads is not reused at 16. The key
    also carries the device index, a power-of-two VRAM class, the CUDA driver / runtime versions and the numba / cupy versions. Adding a
    component changes the key, which only triggers a re-tune; the old per-host directories are simply no longer consulted.

    The hardware part is cached two ways:
      * ``lru_cache(maxsize=1)`` for the process lifetime.
      * On-disk JSON at ``<cache_dir>/.hw_fingerprint.json`` shared across processes (7-day freshness, entries keyed by the visible-device
        selector, re-validated against the installed numba / cupy versions; delete the file or set ``PYUTILZ_HW_FP_REFRESH=1`` to force a
        re-probe after a driver or GPU swap). The thread count is appended after the read, never persisted.

    The cross-process cache exists because ``_cpu_model_slug()`` calls ``cpuinfo.get_cpu_info()`` (~1.9s cold on Windows) and the GPU probe
    queries nvidia-smi (~100ms-2s cold).

    Only a definite answer is persisted: a failed probe (``gpu-unknown``) and an opted-out run (``CUDA_VISIBLE_DEVICES`` empty,
    ``PYUTILZ_DISABLE_GPU=1`` or a registered opt-out predicate) are keyed in memory only, and an opted-out run does not read the GPU
    entries either, so it resolves the CPU-only key of the hardware it will actually use. A host where the stack reports zero devices without
    error persists ``no-gpu`` as before.
    """
    opted_out = _gpu_opted_out()
    threads = f"_t{_numba_threads()}"
    if not opted_out:
        disk = _read_hw_fingerprint_from_disk()
        if disk is not None:
            return disk + threads
    # Resolve the two HW probes through the FACADE package
    # (``pyutilz.performance.kernel_tuning.cache``) rather than this submodule so a
    # ``monkeypatch.setattr(cache, "_cpu_model_slug", ...)`` on the public package --
    # as every kernel-tuning test does -- is honored here. The facade re-exports the
    # real functions from this module by default; late attribute lookup means a patch
    # applied to the package is seen, while an unpatched run calls the originals.
    import sys as _sys
    _facade = _sys.modules.get("pyutilz.performance.kernel_tuning.cache")
    _cpu_probe = getattr(_facade, "_cpu_model_slug", _cpu_model_slug)
    _gpu_probe = getattr(_facade, "_gpu_slug_and_cc", _gpu_slug_and_cc)
    cpu = _cpu_probe()
    gpu, cc = (_GPU_NONE, "") if opted_out else _gpu_probe()
    stamp = _package_stamp()
    if gpu == _GPU_NONE:
        base = f"cpu_{cpu}_{_GPU_NONE}_{stamp}"
    elif gpu == _GPU_UNKNOWN:
        base = f"cpu_{cpu}_{_GPU_UNKNOWN}_{stamp}"
    else:
        try:
            identity = _gpu_identity_suffix()
        except Exception as e:
            logger.warning("GPU identity probe failed (%s), key carries no device details", e)
            identity = "dx"
        base = f"cpu_{cpu}_gpu_{gpu}_cc{cc}_{identity}_{stamp}"
    if not opted_out and gpu != _GPU_UNKNOWN:
        _write_hw_fingerprint_to_disk(base)
    return base + threads


@lru_cache(maxsize=8)
def _ensure_cache_dir(path: str) -> str:
    """``os.makedirs(path, exist_ok=True)`` -- but at most ONCE per distinct path per process.

    After the first call the directory provably exists for the process lifetime, yet the syscall was
    being re-issued on every ``cache_dir()`` / ``host_cache_dir()`` call: measured on this box
    ``makedirs(exist_ok=True)`` on an existing path costs 238 us against 47 us for a bare
    ``isdir`` -- and ``host_cache_dir()`` issues two of them. Memoizing the CREATION rather than
    switching to exists-then-skip keeps ``exist_ok``'s race tolerance (which is why it was chosen)
    on the one call that actually creates the directory. Tests that repoint
    ``PYUTILZ_KERNEL_CACHE_DIR`` and expect the new directory to be created must call
    ``_ensure_cache_dir.cache_clear()`` -- a different path is a different key, so this only matters
    when a path is deleted out from under the process.
    """
    os.makedirs(path, exist_ok=True)
    return path


def cache_dir() -> str:
    """Resolve the on-disk cache directory.

    Order:
        1. ``$PYUTILZ_KERNEL_CACHE_DIR`` env var, if set.
        2. ``~/.pyutilz/kernel_tuning/`` default.

    Creates the directory on first call.
    """
    override = os.environ.get("PYUTILZ_KERNEL_CACHE_DIR", "").strip()
    if override:
        path = override
    else:
        path = os.path.join(os.path.expanduser("~"), ".pyutilz", "kernel_tuning")
    return _ensure_cache_dir(path)


def cache_path() -> str:
    """Path to the LEGACY monolithic per-host JSON file.

    Kept for backward compatibility (migration source + a stable public path).
    The v3 storage no longer writes this file; tunings live as immutable
    per-kernel files under :func:`host_cache_dir`. ``cache_path`` still resolves
    to the v1/v2 location so a pre-existing monolith is found + migrated.
    """
    return os.path.join(cache_dir(), f"{hw_fingerprint()}.json")


def host_cache_dir() -> str:
    """Per-host directory holding the immutable per-kernel tuning files (v3).

    Layout: ``<cache_dir>/<hw_fingerprint>/<kernel_slug>/<...>.json``. Created on
    first call.
    """
    return _ensure_cache_dir(os.path.join(cache_dir(), hw_fingerprint()))


def _legacy_kernel_dir(host_dir: str, kernel_name: str) -> str:
    """Pre-2026-09-26 kernel directory: the lossy ``_slug`` of the name, shared by names such as ``hist.cpu``/``hist``,
    ``a@b``/``a@c`` or ``Mm``/``mm``. Still READ (every file carries its ``kernel_name``) and cleaned by evict, never
    written."""
    return os.path.join(host_dir, _slug(kernel_name, maxlen=80))


def _kernel_dir(host_dir: str, kernel_name: str) -> str:
    """Directory for one kernel's immutable tuning files: ``<readable-part>-<blake2b(kernel_name)[:16 hex]>``.

    The readable part keeps case-insensitive-safe characters only and is informational; the digest of the EXACT name
    makes the directory unique per kernel, so two names can no longer share a directory (and its GC budget).
    """
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", kernel_name).strip("._-").lower()[:64]
    if not readable:
        readable = "kernel"  # a name with no filename-safe character; the digest still makes the dir unique
    digest = hashlib.blake2b(kernel_name.encode("utf-8"), digest_size=8).hexdigest()
    return os.path.join(host_dir, f"{readable}-{digest}")


def _sweep_budget_seconds() -> float:
    """Max-sweep budget (seconds) after which an INPROGRESS marker is steal-able."""
    raw = os.environ.get("PYUTILZ_KERNEL_SWEEP_BUDGET_SEC", "").strip()
    if raw:
        try:
            return max(1.0, float(raw))
        except ValueError:
            pass
    return _DEFAULT_SWEEP_BUDGET_SECONDS


# Windows kernel32 handle for the liveness probe, built lazily ONCE (a ``ctypes.WinDLL``
# construction costs 16 us of the 52 us probe, and the handle is a process-lifetime constant).
# Module-level ``None`` rather than an import-time build so importing this module on a non-Windows
# host never touches ``WinDLL``, which does not exist there.
_KERNEL32 = None


def _kernel32():
    """The shared ``kernel32`` handle, created on first use.

    Must be built with ``use_last_error=True``: the stock ``ctypes.windll.kernel32`` does NOT
    capture the Win32 thread-local last error, so ``ctypes.get_last_error()`` would read 0
    (uninitialized) after a failed ``OpenProcess`` -- a dead pid then reported ALIVE and a stale
    sweep marker owned by a crashed process was never stolen (the sweep wedged). The last-error
    value itself is thread-local and read per call, so SHARING the library handle is safe.
    """
    global _KERNEL32
    if _KERNEL32 is None:
        import ctypes

        _KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]  # Windows-only: ctypes.windll / ctypes.WinDLL are absent on the Linux CI runner, where this same line is a genuine attr-defined
    return _KERNEL32


def _pid_alive(pid: int) -> bool:
    """Best-effort: is ``pid`` a live process on this host? Conservative -- on any
    probe uncertainty returns True (don't steal a marker we can't prove is dead)."""
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            # No os.kill(pid, 0) signal semantics on Windows; query the OS task list, through the
            # shared use_last_error=True handle (see ``_kernel32``).
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            kernel32 = _kernel32()
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                # ERROR_INVALID_PARAMETER (87) => no such pid (dead). Any other failure
                # (e.g. ERROR_ACCESS_DENIED 5 => alive but not ours) => assume alive.
                return ctypes.get_last_error() not in (87,)  # type: ignore[attr-defined]  # Windows-only: ctypes.get_last_error is declared only under the Windows stubs
            try:
                exit_code = ctypes.c_ulong()
                if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    STILL_ACTIVE = 259
                    return exit_code.value == STILL_ACTIVE
                return True
            finally:
                kernel32.CloseHandle(handle)
        else:
            os.kill(pid, 0)
            return True
    except OSError as e:
        # ESRCH => dead; EPERM => alive but not ours.
        return getattr(e, "errno", None) != errno.ESRCH
    except Exception as e:
        logger.debug("Process-liveness probe for pid=%s failed unexpectedly (%s), assuming alive (never steal)", pid, e)
        return True  # never steal on an unexpected probe failure


# ---------------------------------------------------------------------------
# Provenance (recorded on every save; readable for staleness checks)
# ---------------------------------------------------------------------------

def _safe_version(import_name: str, attr: str = "__version__") -> Optional[str]:
    """Return module's version string or None if module / attr is missing."""
    try:
        mod = __import__(import_name)
        return str(getattr(mod, attr, None))
    except Exception as e:
        logger.debug("Could not read %s.%s (%s), version omitted from provenance", import_name, attr, e)
        return None


@lru_cache(maxsize=1)
def _build_provenance_cached() -> dict:
    """Inner memoized worker for :func:`_build_provenance` -- see its docstring. Returns the SHARED
    snapshot; never hand this object to a caller."""
    prov: dict[str, object] = {
        "python_version": "%d.%d" % __import__("sys").version_info[:2],
        "numpy_version": _safe_version("numpy"),
        "numba_version": _safe_version("numba"),
        "cupy_version": _safe_version("cupy"),
    }
    # CUDA runtime + driver (if cupy is importable).
    try:
        import cupy as cp  # type: ignore[import-not-found]  # optional dependency, absent in a minimal install
        try:
            prov["cuda_runtime_version"] = int(cp.cuda.runtime.runtimeGetVersion())
        except Exception as e:  # nosec B110 - best-effort provenance field; a missing/failing CUDA runtime query must not break cache save, provenance dict just omits the field
            logger.debug("Could not read cuda_runtime_version for provenance: %s", e)
            pass
        try:
            prov["cuda_driver_version"] = int(cp.cuda.runtime.driverGetVersion())
        except Exception as e:  # nosec B110 - best-effort provenance field; a missing/failing CUDA driver query must not break cache save, provenance dict just omits the field
            logger.debug("Could not read cuda_driver_version for provenance: %s", e)
            pass
    except ImportError:
        pass
    # Live GPU capability summary (cc, vram, name) at save time.
    try:
        summary = _gpu_summary_cached(_current_device_id())
        if summary is not None:
            prov["gpu_summary"] = {
                "cc_major": summary.get("cc_major"),
                "cc_minor": summary.get("cc_minor"),
                "name": summary.get("name"),
                "total_vram_gb": summary.get("total_vram_gb"),
                "sm_count": summary.get("sm_count"),
            }
    except Exception as e:  # nosec B110 - best-effort provenance enrichment (GPU cc/vram/name summary); failure must not block cache save, provenance dict just omits gpu_summary
        logger.debug("Could not build gpu_summary for provenance: %s", e)
        pass
    return prov


def _build_provenance() -> dict:
    """Snapshot of the env that produced this tuning. Recorded on save.
    Readers can compare this dict to the live env and invalidate if
    something material changed (CUDA driver bump, cupy upgrade, etc.).

    Memoized for the process lifetime (9.9/9.4/7.6 us per build): it depends only on installed
    package versions and the CUDA driver, both invariant while the process runs, and the GPU half of
    it is already memoized via ``_gpu_summary_cached``. It was being rebuilt once per KERNEL
    DIRECTORY inside the cache load loop, so a 100-kernel host cache paid ~1 ms of redundant version
    probing at every process start. A shallow copy is returned (with the nested ``gpu_summary``
    copied too) so a caller mutating the snapshot cannot poison every later reader. Tests that
    change the environment must call ``_build_provenance_cached.cache_clear()``.
    """
    prov = dict(_build_provenance_cached())
    gpu = prov.get("gpu_summary")
    if isinstance(gpu, dict):
        prov["gpu_summary"] = dict(gpu)
    return prov


def provenance_changed(old: Optional[dict], new: Optional[dict]) -> bool:
    """True iff a MATERIAL provenance field differs (cuda driver/runtime,
    cupy/numba/numpy versions, GPU cc/name, or python MAJOR.MINOR). python
    major.minor IS material: numba/cupy codegen can differ across interpreter
    minors, so a Python upgrade should re-tune deterministically (cheap +
    correct) rather than let codegen drift invalidate unpredictably. Patch
    bumps are NOT material (python_version stores only major.minor)."""
    if old is None or new is None:
        return False  # be conservative: no data -> no invalidation
    keys = ("cuda_driver_version", "cuda_runtime_version", "cupy_version", "numba_version", "numpy_version", "python_version")
    for k in keys:
        if old.get(k) != new.get(k):
            return True
    old_gpu = old.get("gpu_summary") or {}
    new_gpu = new.get("gpu_summary") or {}
    for k in ("cc_major", "cc_minor", "name"):
        a, b = old_gpu.get(k), new_gpu.get(k)
        # A field one side could not read is unknown, not different: the GPU name comes from a slower probe than cc,
        # and a process that lost that race under load saved name=None, so every reader discarded a valid tuning.
        if a is not None and b is not None and a != b:
            return True
    return False


# ---------------------------------------------------------------------------
# GPU-busy gate for async sweeps
# ---------------------------------------------------------------------------

# An async (fit-time) sweep that runs while the GPU is busy is doubly wrong: it
# (a) contends with the caller's own fit (measured ~18% wall tax on a 100k MRMR
# fit -- 151s vs 124s) and (b) records CONTENDED kernel timings as this host's
# "optimum", which then mis-route every future dispatch. Auto-tuning is only
# valid on an idle GPU, so we defer the sweep when the GPU is loaded and let the
# offline CLI (or a later idle process) tune instead. The shipped default cache
# covers correctness in the meantime. GPUtil.getGPUs() costs ~0.1-2s (nvidia-smi),
# so cache the verdict process-wide for a short TTL -- a burst of sweep spawns at
# fit start then shares ONE poll instead of one per kernel.
# Seconds to wait after deciding a sweep is needed before actually starting it: lets the triggering
# fit get past its bursty start (kernel launches, H2D) so the busy-check below sees the real load, and
# avoids stealing the device the instant the caller needs it. Env-overridable; 0 disables the delay.
def _async_sweep_start_delay() -> float:
    """Seconds to wait after deciding an async sweep is needed before starting it (env ``PYUTILZ_KERNEL_SWEEP_START_DELAY``, default 10s)."""
    try:
        return max(0.0, float(os.environ.get("PYUTILZ_KERNEL_SWEEP_START_DELAY", "10.0")))
    except ValueError:
        return 10.0


def _async_sweep_idle_max_wait() -> float:
    """Max seconds the async sweep waits for the hardware to go idle before sweeping ANYWAY (rather than
    abandoning and leaving the per-host cache empty forever). Env PYUTILZ_KERNEL_SWEEP_IDLE_MAX_WAIT,
    default 120s. 0 -> proceed immediately (no idle wait)."""
    try:
        return max(0.0, float(os.environ.get("PYUTILZ_KERNEL_SWEEP_IDLE_MAX_WAIT", "120.0")))
    except ValueError:
        return 120.0


def _async_sweep_hw_busy() -> bool:
    """True iff the CPU or GPU is busy enough that an async sweep should defer (so we only ever
    benchmark on idle hardware -- a contended sweep both taxes the caller and records contended
    timings as this host's optimum). Delegates to the shared ``benchmark.hardware_busy`` (CPU via
    psutil, GPU via GPUtil, threshold ``PYUTILZ_KERNEL_SWEEP_HW_BUSY`` default 0.40). Never trips on
    hardware it cannot measure (no psutil / no GPU)."""
    try:
        from ..benchmark import hardware_busy
        return hardware_busy()
    except Exception as e:
        logger.debug("hardware_busy() probe failed (%s), treating hardware as idle", e)
        return False

# Process-scoped "tuned this run" guard, keyed on (kernel_name, cache_path), so
# get_or_tune sweeps at most once per kernel per process (tests that switch
# PYUTILZ_KERNEL_CACHE_DIR get a different path -> a fresh re-tune).
_TUNED_THIS_PROCESS: set = set()

# Guards the check-then-add on _TUNED_THIS_PROCESS in get_or_tune(): without it, two threads
# racing for the same (kernel, cache-path) can both observe "not yet tuned" and both spawn a
# redundant async sweep thread (or both proceed into the synchronous sweep-claim path).
_tuned_guard_lock = threading.Lock()

# Process-scoped "already logged the invalidation banner for this kernel" guard, keyed
# exactly like _TUNED_THIS_PROCESS on (kernel_name, cache_path). get_or_tune re-evaluates
# code_version staleness on EVERY call (so a fresh entry an async sweep lands mid-process is
# picked up immediately -- once_per_process only gates the SWEEP), but while a kernel stays
# stale (e.g. a no-op tuner that never persists a fresh entry) that would otherwise fire the
# INFO "invalidated...will re-tune" log on every single call. Log it at most once per kernel
# per process instead; the staleness re-check itself stays unconditional (a cheap dict lookup).
_INVALIDATION_LOGGED_THIS_PROCESS: set = set()

# Process-scoped "already logged a DEFAULT-cache fallback for this kernel" guard, keyed like
# _TUNED_THIS_PROCESS on (kernel_name, cache_path). _fb() (the get_or_tune fallback path) runs
# on EVERY call once the local per-host lookup misses -- for a FIT-TIME dispatcher (async_sweep=True)
# that is every single caller invocation until the background sweep lands (a boosting round's
# per-iteration monitor metric can call this hundreds of times per fit), and each of those iterations
# hits one of the same handful of DEFAULT-cache branches (None / same-instance / stale / no-match /
# raised) for the SAME (kernel_name, dims-shape) reason. Log each distinct (kernel_name, branch) at
# most once per process instead of once per call; the fallback VALUE returned is unaffected, only the
# warning volume is throttled.

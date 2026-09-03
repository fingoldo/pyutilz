"""Scanner tests for async_primitive_reinit, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_async_primitive_reinit_per_call,
)

from ._helpers import _write

# ---- async_primitive_reinit_per_call --------------------------------------


def test_async_primitive_reinit_lock_inside_function_flagged(tmp_path: Path):
    _write(tmp_path, "mod.py", '''
import asyncio

async def handle():
    lock = asyncio.Lock()
    async with lock:
        pass
''')
    findings = scan_async_primitive_reinit_per_call(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "async_primitive_reinit_per_call"
    assert findings[0].severity == "P1"


def test_async_primitive_reinit_semaphore_default_arg_flagged(tmp_path: Path):
    """A primitive constructed as a default-argument expression inside the function body is also flagged."""
    _write(tmp_path, "mod.py", '''
import asyncio

async def handle():
    sem = asyncio.Semaphore(3)
    async with sem:
        pass
''')
    findings = scan_async_primitive_reinit_per_call(tmp_path)
    assert len(findings) == 1


def test_async_primitive_reinit_module_scope_is_clean(tmp_path: Path):
    """A primitive created at module scope (the correct pattern) is not flagged."""
    _write(tmp_path, "mod.py", '''
import asyncio

_LOCK = asyncio.Lock()

async def handle():
    async with _LOCK:
        pass
''')
    assert scan_async_primitive_reinit_per_call(tmp_path) == []


def test_async_primitive_reinit_init_attribute_is_clean(tmp_path: Path):
    """A primitive assigned to self in __init__ (created once per instance, shared across calls) is not flagged."""
    _write(tmp_path, "mod.py", '''
import asyncio

class Worker:
    def __init__(self):
        self._lock = asyncio.Lock()

    async def handle(self):
        async with self._lock:
            pass
''')
    assert scan_async_primitive_reinit_per_call(tmp_path) == []


def test_async_primitive_reinit_global_lazy_singleton_is_clean(tmp_path: Path):
    """The global-declared lazy-module-singleton idiom is the safe shared-instance case."""
    _write(tmp_path, "mod.py", '''
import asyncio

_sem = None

async def get_sem():
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(5)
    return _sem
''')
    assert scan_async_primitive_reinit_per_call(tmp_path) == []


def test_async_primitive_reinit_non_primitive_call_is_clean(tmp_path: Path):
    """An asyncio call that is NOT one of the coordination primitives (e.g. asyncio.sleep) is not flagged."""
    _write(tmp_path, "mod.py", '''
import asyncio

async def handle():
    await asyncio.sleep(0.1)
''')
    assert scan_async_primitive_reinit_per_call(tmp_path) == []


def test_async_primitive_reinit_module_level_registry_is_clean(tmp_path: Path):
    """Publishing a primitive INTO a module-level container is the single-flight
    idiom -- every caller finds the same Event through the shared dict, which is
    the opposite of a private per-call copy. Needs no `global` (never rebound)."""
    _write(tmp_path, "mod.py", '''
import asyncio

_inflight: dict = {}
_inflight_lock = asyncio.Lock()

async def cached_get(key):
    async with _inflight_lock:
        if key not in _inflight:
            _inflight[key] = asyncio.Event()
            return None
        evt = _inflight[key]
    await evt.wait()
''')
    assert scan_async_primitive_reinit_per_call(tmp_path) == []


def test_async_primitive_reinit_local_dict_registry_is_still_flagged(tmp_path: Path):
    """Guard on the exemption above: a FUNCTION-LOCAL dict is not shared, so
    publishing into it is still a private per-call copy and stays flagged."""
    _write(tmp_path, "mod.py", '''
import asyncio

async def cached_get(key):
    inflight = {}
    inflight[key] = asyncio.Event()
    await inflight[key].wait()
''')
    assert len(scan_async_primitive_reinit_per_call(tmp_path)) == 1


def test_async_primitive_reinit_bounded_gather_closure_is_clean(tmp_path: Path):
    """The bounded-gather idiom: the semaphore bounds the tasks THIS call
    spawns, so one per call is correct and deliberate."""
    _write(tmp_path, "mod.py", '''
import asyncio

async def bounded_gather(factories, limit):
    sem = asyncio.Semaphore(limit)

    async def _run(factory):
        async with sem:
            return await factory()

    return await asyncio.gather(*[_run(f) for f in factories])
''')
    assert scan_async_primitive_reinit_per_call(tmp_path) == []


def test_async_primitive_reinit_handed_to_helper_is_clean(tmp_path: Path):
    """Same idiom written with a helper instead of a closure."""
    _write(tmp_path, "mod.py", '''
import asyncio

async def run_round(session, payload):
    sem = asyncio.Semaphore(4)
    return await _run_pipeline_for(session, sem, payload)
''')
    assert scan_async_primitive_reinit_per_call(tmp_path) == []


def test_async_primitive_reinit_direct_use_still_flagged_alongside_a_closure(tmp_path: Path):
    """Non-vacuousness guard for BOTH fan-out exemptions: a function that also
    defines a closure must not become a blanket amnesty. The lock here is used
    directly in the body and never reaches the closure, so it stays flagged."""
    _write(tmp_path, "mod.py", '''
import asyncio

async def handle(items):
    lock = asyncio.Lock()

    async def _work(i):
        return i * 2

    async with lock:
        pass
    return await asyncio.gather(*[_work(i) for i in items])
''')
    findings = scan_async_primitive_reinit_per_call(tmp_path)
    assert len(findings) == 1, [f.detail for f in findings]


def test_async_primitive_reinit_custom_primitive_names(tmp_path: Path):
    """The primitive_names parameter can narrow/widen which asyncio.* constructors are tracked."""
    _write(tmp_path, "mod.py", '''
import asyncio

async def handle():
    lock = asyncio.Lock()
    async with lock:
        pass
''')
    assert scan_async_primitive_reinit_per_call(tmp_path, primitive_names=frozenset({"Event"})) == []

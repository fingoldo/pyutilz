"""Scanner tests for missed_await, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_missed_await,
    scan_sync_blocking_in_async,
)

from ._helpers import _write

# ---- missed_await ----------------------------------------------------------


def test_missed_await_discarded_coroutine_flags(tmp_path: Path):
    """The true-positive shape: a bare-statement call to a same-module
    async def -- the coroutine is created and discarded, the body never
    runs, and the caller carries on as if the save happened."""
    _write(tmp_path, "bad.py", """
async def do_save(item):
    ...

async def process(item):
    do_save(item)
    return True
""")
    findings = scan_missed_await(tmp_path)
    assert len(findings) == 1
    assert findings[0].severity == "P1"
    assert "do_save" in findings[0].detail


def test_missed_await_from_sync_caller_also_flags(tmp_path: Path):
    """Discarding a coroutine from a SYNC function is the same bug."""
    _write(tmp_path, "bad.py", """
async def notify(msg):
    ...

def handler(msg):
    notify(msg)
""")
    assert scan_missed_await(tmp_path), "sync caller discarding a coroutine must be flagged"


def test_missed_await_awaited_call_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
async def do_save(item):
    ...

async def process(item):
    await do_save(item)
""")
    assert scan_missed_await(tmp_path) == []


def test_missed_await_gather_list_pattern_clean(tmp_path: Path):
    """FP shape #1 from corpus validation: coroutines collected into a
    list and gathered later -- assignment-level calls are never flagged."""
    _write(tmp_path, "ok.py", """
import asyncio

async def work(x):
    ...

async def run(xs):
    tasks = [work(x) for x in xs]
    return await asyncio.gather(*tasks)
""")
    assert scan_missed_await(tmp_path) == []


def test_missed_await_local_import_shadow_clean(tmp_path: Path):
    """FP shape #2 from corpus validation: a function-local import of a
    SYNC function that shares its name with a module-level async def."""
    _write(tmp_path, "ok.py", """
async def count_tokens(text):
    ...

def fallback(text):
    from other_module import count_tokens
    count_tokens(text)
""")
    assert scan_missed_await(tmp_path) == []


def test_missed_await_local_assignment_shadow_clean(tmp_path: Path):
    """FP shape #3: the name is locally rebound to something else."""
    _write(tmp_path, "ok.py", """
async def refresh():
    ...

def run(callbacks):
    refresh = callbacks["refresh"]
    refresh()
""")
    assert scan_missed_await(tmp_path) == []


def test_missed_await_attribute_call_not_flagged(tmp_path: Path):
    """Attribute calls (self.method(), obj.fn()) are out of scope -- no
    reliable static resolution to a same-module async def."""
    _write(tmp_path, "ok.py", """
class Svc:
    async def ping(self):
        ...

    def run(self):
        self.ping()
""")
    assert scan_missed_await(tmp_path) == []


# ---- sync_blocking_in_async --------------------------------------------


def test_sync_blocking_in_async_flags_bare_requests(tmp_path: Path):
    _write(tmp_path, "bad.py", """
import requests

async def generate():
    return requests.get("http://example.com")
""")
    findings = scan_sync_blocking_in_async(tmp_path)
    assert len(findings) == 1
    assert findings[0].severity == "P1"


def test_sync_blocking_in_async_awaited_httpx_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
import httpx

async def generate():
    async with httpx.AsyncClient() as client:
        return await client.get("http://example.com")
""")
    findings = scan_sync_blocking_in_async(tmp_path)
    assert findings == []


# ---- F206: missed await at module scope ---------------------------------------------


def test_missed_await_sees_a_module_level_call(tmp_path: Path):
    _write(tmp_path, "a.py", """
async def main():
    pass

if __name__ == "__main__":
    main()
""")
    findings = scan_missed_await(tmp_path)
    assert len(findings) == 1 and findings[0].line == 5


def test_missed_await_accepts_an_awaited_module_level_call(tmp_path: Path):
    _write(tmp_path, "a.py", """
import asyncio

async def main():
    pass

if __name__ == "__main__":
    asyncio.run(main())
""")
    assert scan_missed_await(tmp_path) == []

"""Scanner tests for shielded_resource_release, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_shielded_resource_release_race,
)

from ._helpers import _write

# ---- shielded_resource_release_race ---------------------------------------


def test_shielded_resource_release_race_flagged(tmp_path: Path):
    _write(tmp_path, "bad.py", """
import asyncio

async def save_and_notify(pool):
    conn = pool.acquire()
    try:
        async def _do_work():
            await conn.execute("insert ...")
        await asyncio.shield(_do_work())
    finally:
        release_conn(conn)
""")
    findings = scan_shielded_resource_release_race(tmp_path)
    assert len(findings) == 1, findings
    assert findings[0].check == "shielded_resource_release_race"
    assert findings[0].severity == "P0"


def test_shielded_resource_release_race_own_resource_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
import asyncio

async def save_and_notify(pool):
    conn = pool.acquire()
    try:
        async def _do_work():
            own_conn = pool.acquire()
            try:
                await own_conn.execute("insert ...")
            finally:
                release_conn(own_conn)
        await asyncio.shield(_do_work())
    finally:
        release_conn(conn)
""")
    findings = scan_shielded_resource_release_race(tmp_path)
    assert findings == []


def test_shielded_resource_release_race_no_shield_is_clean(tmp_path: Path):
    _write(tmp_path, "ok.py", """
import asyncio

async def save_and_notify(pool):
    conn = pool.acquire()
    try:
        async def _do_work():
            await conn.execute("insert ...")
        await _do_work()
    finally:
        release_conn(conn)
""")
    findings = scan_shielded_resource_release_race(tmp_path)
    assert findings == []


def test_shielded_resource_release_race_custom_release_names(tmp_path: Path):
    _write(tmp_path, "bad.py", """
import asyncio

async def save_and_notify(pool):
    conn = pool.acquire()
    try:
        async def _do_work():
            await conn.execute("insert ...")
        await asyncio.shield(_do_work())
    finally:
        my_custom_release(conn)
""")
    assert scan_shielded_resource_release_race(tmp_path) == []
    findings = scan_shielded_resource_release_race(tmp_path, release_call_names=frozenset({"my_custom_release"}))
    assert len(findings) == 1

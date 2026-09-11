"""Scanner tests for dead_endpoint_parameter: the glossum `strategy` shape, each binding style, each exclusion.

Every test calls ``scan_dead_endpoint_parameters`` by name and asserts on what it returned: the polarity gate in
``tests/test_meta/test_scanner_positive_and_negative_cases.py`` reads those assertions out of this source, and a
call hidden behind a local helper is invisible to it.
"""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import scan_dead_endpoint_parameters

from ._helpers import _write

_HEAD = """
from typing import Annotated, Literal
from fastapi import APIRouter, Body, Depends, Header, Path, Query, Request, Security
router = APIRouter()
"""


def test_the_strategy_shape_is_flagged(tmp_path: Path):
    """Declared, described in the schema, named in the docstring, read nowhere: the 05-F3 / 11-F4 case."""
    _write(tmp_path, "routes.py", _HEAD + '''
@router.get("/batch/learning")
async def batch(limit: int = Query(20), strategy: Literal["random", "easy", "hard"] = Query(default="random", description="Selection strategy")):
    """Returns cards; strategy picks random, easy or hard."""
    return {"limit": limit}
''')

    findings = scan_dead_endpoint_parameters(tmp_path)

    assert [(f.check, f.severity) for f in findings] == [("dead_endpoint_parameter", "P2")]
    assert "`strategy`" in findings[0].detail and "(Query)" in findings[0].detail


def test_a_parameter_that_is_read_is_clean(tmp_path: Path):
    """A parameter read in the body, including from a nested function, is alive."""
    _write(tmp_path, "routes.py", _HEAD + '''
@router.get("/words/{word_id}")
async def word(word_id: str = Path(...), lang: str = Query("en")):
    def inner():
        return lang
    return {"id": word_id, "lang": inner()}
''')

    assert scan_dead_endpoint_parameters(tmp_path) == []


def test_the_annotated_style_is_in_the_population(tmp_path: Path):
    _write(tmp_path, "routes.py", _HEAD + '''
@router.post("/track")
async def track(event: Annotated[str, Body()], trace: Annotated[str | None, Header()] = None):
    return {"event": event}
''')

    findings = scan_dead_endpoint_parameters(tmp_path)

    assert [f.line for f in findings] == [6]
    assert "`trace`" in findings[0].detail


def test_what_is_not_in_the_population(tmp_path: Path):
    """A dependency wanted for its side effect, an underscore name, Request, and a function that is not a route."""
    _write(tmp_path, "routes.py", _HEAD + '''
def get_db():
    yield None

@router.get("/ready")
async def ready(request: Request, db=Depends(get_db), user=Security(get_db), _debug: bool = Query(False)):
    return {"ok": True}

def helper(q: str = Query("x")):
    return None
''')

    assert scan_dead_endpoint_parameters(tmp_path) == []


def test_a_websocket_and_api_route_are_routes(tmp_path: Path):
    _write(tmp_path, "routes.py", _HEAD + '''
@router.websocket("/ws")
async def ws(token: str = Query(...)):
    return None

@router.api_route("/any", methods=["GET", "POST"])
async def anything(mode: str = Query("a")):
    return None
''')

    findings = scan_dead_endpoint_parameters(tmp_path)

    assert sorted(f.detail.split("`")[3] for f in findings) == ["mode", "token"]

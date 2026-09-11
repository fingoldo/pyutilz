"""Scanner tests for signature_gated_kwarg_drop: the glossum LLMClient shape, and each exclusion.

Every test calls ``scan_signature_gated_kwarg_drops`` by name and asserts on what it returned: the polarity gate in
``tests/test_meta/test_scanner_positive_and_negative_cases.py`` reads those assertions out of this source.
"""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import scan_signature_gated_kwarg_drops

from ._helpers import _write

_HEAD = """
import inspect
import logging
logger = logging.getLogger(__name__)
"""


def test_the_silent_drop_is_flagged(tmp_path: Path):
    """The proposal #6 shape: sent where the provider declares it, gone everywhere else."""
    _write(tmp_path, "client.py", _HEAD + '''
async def call(provider, prompt, thinking):
    gen_kwargs = {"prompt": prompt}
    params = set(inspect.signature(provider.generate).parameters)
    if "thinking" in params:
        gen_kwargs["thinking"] = thinking
    return await provider.generate(**gen_kwargs)
''')

    findings = scan_signature_gated_kwarg_drops(tmp_path)

    assert [(f.check, f.severity) for f in findings] == [("signature_gated_kwarg_drop", "P2")]
    assert "`thinking`" in findings[0].detail
    assert "**kwargs" in findings[0].detail


def test_the_inline_signature_call_is_the_same_shape(tmp_path: Path):
    """No intermediate name: the membership test reads the signature where it stands."""
    _write(tmp_path, "client.py", _HEAD + '''
def call(provider, temperature, gen_kwargs):
    if "temperature" in inspect.signature(provider.generate).parameters:
        gen_kwargs["temperature"] = temperature
    return gen_kwargs
''')

    findings = scan_signature_gated_kwarg_drops(tmp_path)

    assert [f.line for f in findings] == [6]


def test_a_try_except_fallback_still_binds_the_parameter_names(tmp_path: Path):
    """The defensive form, where an unreadable signature falls back to an empty set."""
    _write(tmp_path, "client.py", _HEAD + '''
def call(provider, json_mode, gen_kwargs):
    try:
        params = set(inspect.signature(provider.generate).parameters)
    except (ValueError, TypeError):
        params = set()
    if "json_mode" in params:
        gen_kwargs["json_mode"] = json_mode
    return gen_kwargs
''')

    findings = scan_signature_gated_kwarg_drops(tmp_path)

    assert [f.line for f in findings] == [10]


def test_a_reported_drop_and_other_membership_tests_are_clean(tmp_path: Path):
    """The fix (an else that says so), a raising else, and a membership test that is not about a signature."""
    _write(tmp_path, "client.py", _HEAD + '''
def warned(provider, thinking, gen_kwargs):
    params = set(inspect.signature(provider.generate).parameters)
    if "thinking" in params:
        gen_kwargs["thinking"] = thinking
    else:
        logger.warning("%s takes no thinking parameter; %r was not sent", provider, thinking)
    return gen_kwargs

def refused(provider, thinking, gen_kwargs):
    params = set(inspect.signature(provider.generate).parameters)
    if "thinking" in params:
        gen_kwargs["thinking"] = thinking
    else:
        raise TypeError("provider takes no thinking")
    return gen_kwargs

def unrelated(payload, defaults):
    if "thinking" in payload:
        defaults["thinking"] = payload["thinking"]
    return defaults
''')

    assert scan_signature_gated_kwarg_drops(tmp_path) == []


def test_a_gate_that_forwards_nothing_is_not_a_drop(tmp_path: Path):
    """Branching on what a callee accepts is fine; only losing the caller's value is the defect."""
    _write(tmp_path, "client.py", _HEAD + '''
def call(provider):
    params = set(inspect.signature(provider.generate).parameters)
    if "thinking" in params:
        return "supported"
    return "unsupported"
''')

    assert scan_signature_gated_kwarg_drops(tmp_path) == []

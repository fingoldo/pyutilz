"""Scanner tests for llm_max_tokens_cap, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import (
    scan_llm_call_missing_max_tokens_cap,
)

from ._helpers import _write

# ---- llm_call_missing_max_tokens_cap ---------------------------------------


def test_llm_max_tokens_cap_missing_kwarg_flagged(tmp_path: Path):
    _write(tmp_path, "mod.py", '''
provider = get_llm_provider("anthropic")
provider.generate("hello")
''')
    findings = scan_llm_call_missing_max_tokens_cap(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "llm_call_missing_max_tokens_cap"
    assert findings[0].severity == "P2"


def test_llm_max_tokens_cap_zero_literal_flagged(tmp_path: Path):
    """An explicit max_tokens=0 is the same as omitting it -- still flagged."""
    _write(tmp_path, "mod.py", '''
provider = get_llm_provider("anthropic")
provider.generate("hello", max_tokens=0)
''')
    findings = scan_llm_call_missing_max_tokens_cap(tmp_path)
    assert len(findings) == 1


def test_llm_max_tokens_cap_generate_json_and_generate_batch_flagged(tmp_path: Path):
    """generate_json and generate_batch are also tracked capped methods."""
    _write(tmp_path, "mod.py", '''
provider = get_llm_provider("anthropic")
provider.generate_json("hello")
provider.generate_batch(["a", "b"])
''')
    findings = scan_llm_call_missing_max_tokens_cap(tmp_path)
    assert len(findings) == 2


def test_llm_max_tokens_cap_explicit_nonzero_is_clean(tmp_path: Path):
    _write(tmp_path, "mod.py", '''
provider = get_llm_provider("anthropic")
provider.generate("hello", max_tokens=2000)
''')
    assert scan_llm_call_missing_max_tokens_cap(tmp_path) == []


def test_llm_max_tokens_cap_non_provider_variable_is_clean(tmp_path: Path):
    """A .generate(...) call on a variable NOT assigned from get_llm_provider(...) is not tracked."""
    _write(tmp_path, "mod.py", '''
other = SomeUnrelatedClass()
other.generate("hello")
''')
    assert scan_llm_call_missing_max_tokens_cap(tmp_path) == []


def test_llm_max_tokens_cap_no_provider_in_module_short_circuits(tmp_path: Path):
    """A module with no get_llm_provider(...) assignment at all is skipped entirely (cheap early-out)."""
    _write(tmp_path, "mod.py", '''
def f():
    return 1
''')
    assert scan_llm_call_missing_max_tokens_cap(tmp_path) == []

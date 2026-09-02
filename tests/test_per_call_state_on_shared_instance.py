"""Unit tests for the ``per_call_state_on_shared_instance`` scanner.

Each defective fixture below is a reduction of a real 2026-09-02 finding, written out to a tmp tree
rather than by breaking the shipped source, and each is paired with the fixed shape that must be
silent - so the test proves both that the rule fires and that the fix is what quiets it.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from pyutilz.dev.code_audit import scan_per_call_state_on_shared_instance

# 09/F08 + 09/F57: OpenRouter's per-call metadata written on the shared instance and read back by a
# summary accessor, with no lock. The registry dict is what marks the class as shared: nothing ever
# writes `_provider_cache[key] = OpenRouterProvider(...)` literally.
_OPENROUTER_DEFECTIVE = '''
_PROVIDER_MODULES = {"openrouter": ("pkg.openrouter", "OpenRouterProvider", "openrouter_api_key")}


class OpenRouterProvider:
    def __init__(self):
        self.last_generation_id = None
        self.last_upstream_provider = None

    async def generate(self, prompt):
        response = await self._post(prompt)
        self.last_generation_id = response["id"]
        self.last_upstream_provider = response["provider"]
        return response["text"]

    async def _post(self, prompt):
        return {"id": "x", "provider": "y", "text": prompt}

    def last_call_summary(self):
        return {"generation_id": self.last_generation_id, "provider": self.last_upstream_provider}
'''

_OPENROUTER_FIXED = '''
_PROVIDER_MODULES = {"openrouter": ("pkg.openrouter", "OpenRouterProvider", "openrouter_api_key")}


class OpenRouterProvider:
    async def generate(self, prompt):
        response = await self._post(prompt)
        return {"text": response["text"], "generation_id": response["id"], "provider": response["provider"]}

    async def _post(self, prompt):
        return {"id": "x", "provider": "y", "text": prompt}
'''

# 09/F09: `_last_json_schema_applied` clobbered across concurrent calls, so the strict-schema
# guarantee flag lies. Declared on the BASE class while only the subclass is registered -- the
# scanner's base-class propagation is what makes this reachable.
_OPENAI_COMPAT_DEFECTIVE = '''
_PROVIDER_MODULES = {"deepseek": ("pkg.deepseek", "DeepSeekProvider", "deepseek_api_key")}


class OpenAICompatibleProvider:
    async def generate(self, prompt, json_schema=None):
        self._last_json_schema_applied = json_schema is not None
        return prompt

    def json_schema_was_applied(self):
        return self._last_json_schema_applied


class DeepSeekProvider(OpenAICompatibleProvider):
    pass
'''

_OPENAI_COMPAT_FIXED = '''
_PROVIDER_MODULES = {"deepseek": ("pkg.deepseek", "DeepSeekProvider", "deepseek_api_key")}


class OpenAICompatibleProvider:
    async def generate(self, prompt, json_schema=None):
        return {"text": prompt, "json_schema_applied": json_schema is not None}


class DeepSeekProvider(OpenAICompatibleProvider):
    pass
'''

# A lock-guarded write on the same shape must NOT be reported.
_LOCK_GUARDED = '''
_PROVIDER_MODULES = {"x": ("pkg.x", "XProvider", "x_api_key")}


class XProvider:
    async def generate(self, prompt):
        async with self._lock:
            self.last_usage = len(prompt)
        return prompt

    def last_call_summary(self):
        return self.last_usage
'''

# A class nobody shares (constructed per use, in no registry) is out of scope by construction.
_NOT_SHARED = '''
class OneShotRequest:
    async def run(self, prompt):
        self.last_usage = len(prompt)
        return prompt

    def last_call_summary(self):
        return self.last_usage
'''


def _scan(tmp_path: Path, name: str, source: str):
    (tmp_path / name).write_text(textwrap.dedent(source), encoding="utf-8")
    return scan_per_call_state_on_shared_instance(tmp_path)


def _attrs(findings) -> set[str]:
    return {f.detail.split("`")[1].replace("self.", "") for f in findings}


def test_flags_per_call_metadata_on_a_registry_shared_provider(tmp_path: Path) -> None:
    findings = _scan(tmp_path, "openrouter.py", _OPENROUTER_DEFECTIVE)
    assert _attrs(findings) == {"last_generation_id", "last_upstream_provider"}
    assert {(f.check, f.severity) for f in findings} == {("per_call_state_on_shared_instance", "P2")}


def test_returning_metadata_per_call_clears_the_finding(tmp_path: Path) -> None:
    assert _scan(tmp_path, "openrouter.py", _OPENROUTER_FIXED) == []


def test_flags_state_declared_on_the_base_of_a_registered_subclass(tmp_path: Path) -> None:
    findings = _scan(tmp_path, "openai_compat.py", _OPENAI_COMPAT_DEFECTIVE)
    assert _attrs(findings) == {"_last_json_schema_applied"}


def test_base_class_fix_clears_the_finding(tmp_path: Path) -> None:
    assert _scan(tmp_path, "openai_compat.py", _OPENAI_COMPAT_FIXED) == []


def test_lock_guarded_write_is_not_flagged(tmp_path: Path) -> None:
    assert _scan(tmp_path, "guarded.py", _LOCK_GUARDED) == []


def test_unshared_class_is_out_of_scope(tmp_path: Path) -> None:
    assert _scan(tmp_path, "oneshot.py", _NOT_SHARED) == []


def test_scanner_is_opt_in_only() -> None:
    """Warn-first: it must not join the default sweep, whose findings block via the baseline test."""
    from pyutilz.dev.code_audit.registry import OPT_IN_ONLY, get_scanners

    assert "per_call_state_on_shared_instance" in get_scanners()
    assert "per_call_state_on_shared_instance" in OPT_IN_ONLY

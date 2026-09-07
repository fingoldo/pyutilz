"""Scanner tests for override_signature_drift, the base method left behind by its own overrides.

Review notes, and the reason nothing here is baselined:

* The rule was measured before it was wired, over ``src/pyutilz``, ``py-ci-shared/src`` and
  ``mlframe/src``. In the sibling-agreement form implemented here it reports zero hits in pyutilz
  and two in mlframe, so pyutilz needs no baseline entry at all -- there is nothing to suppress.
* Three weaker candidates were measured and rejected rather than wired, and the numbers are in the
  report for that change: "an override accepts a parameter its base does not", taken on its own,
  keeps one pyutilz hit that is ordinary specialisation (``OpenAICompatibleProvider.generate_json``
  supports ``json_schema``; no sibling does); "an override drops a parameter its base declares"
  produced one hit across all three trees and it was a base hook's ``_unused``-style rename, i.e.
  zero real; "the same parameter typed differently" produced none anywhere.
* ``test_catches_the_generate_json_thinking_incident`` reconstructs the pre-fix shape of the
  September 2026 defect -- ``thinking`` on three provider overrides and not on ``LLMProvider`` --
  from the same class and method names, so the regression it pins is the real one.
"""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit.override_signature_drift import scan_override_signature_drift

from ._helpers import _write

# ---- override_signature_drift --------------------------------------------


def test_flags_a_base_two_siblings_have_widened(tmp_path: Path):
    _write(
        tmp_path,
        "providers.py",
        """
class Base:
    def run(self, prompt: str, temperature: float = 0.3) -> str:
        return prompt


class First(Base):
    def run(self, prompt: str, temperature: float = 0.3, thinking: bool = False) -> str:
        return prompt


class Second(Base):
    def run(self, prompt: str, temperature: float = 0.3, thinking: bool = False) -> str:
        return prompt
""",
    )
    findings = scan_override_signature_drift(tmp_path)
    assert len(findings) == 1
    assert findings[0].check == "override_signature_drift"
    assert findings[0].file == "providers.py"
    assert "thinking" in findings[0].detail
    assert "First" in findings[0].detail and "Second" in findings[0].detail


def test_catches_the_generate_json_thinking_incident(tmp_path: Path):
    """The real defect: `thinking` on three provider overrides and not on the base they inherit."""
    _write(
        tmp_path,
        "base.py",
        """
class LLMProvider:
    async def generate_json(self, prompt: str, system: str = "", images: list = None) -> dict:
        return {}
""",
    )
    for module, klass in (("claude_code_provider.py", "ClaudeCodeProvider"), ("gemini_provider.py", "GeminiProvider"), ("openai_compat.py", "OpenAICompatibleProvider")):
        _write(
            tmp_path,
            module,
            f"""
from base import LLMProvider


class {klass}(LLMProvider):
    async def generate_json(self, prompt: str, system: str = "", images: list = None, thinking: bool = None) -> dict:
        return {{}}
""",
        )
    findings = scan_override_signature_drift(tmp_path)
    assert [f.file for f in findings] == ["base.py"]
    assert "generate_json" in findings[0].detail and "thinking" in findings[0].detail


def test_one_widening_override_is_specialisation_not_drift(tmp_path: Path):
    _write(
        tmp_path,
        "solo.py",
        """
class Base:
    def run(self, prompt: str) -> str:
        return prompt


class Only(Base):
    def run(self, prompt: str, json_schema: dict = None) -> str:
        return prompt
""",
    )
    assert scan_override_signature_drift(tmp_path) == []


def test_abstract_base_is_exempt(tmp_path: Path):
    _write(
        tmp_path,
        "abstract.py",
        """
from abc import ABC, abstractmethod


class Base(ABC):
    @abstractmethod
    def run(self, prompt: str) -> str:
        ...


class First(Base):
    def run(self, prompt: str, thinking: bool = False) -> str:
        return prompt


class Second(Base):
    def run(self, prompt: str, thinking: bool = False) -> str:
        return prompt
""",
    )
    assert scan_override_signature_drift(tmp_path) == []


def test_not_implemented_stub_base_is_exempt(tmp_path: Path):
    _write(
        tmp_path,
        "stub.py",
        """
class Base:
    def run(self, prompt: str) -> str:
        raise NotImplementedError


class First(Base):
    def run(self, prompt: str, thinking: bool = False) -> str:
        return prompt


class Second(Base):
    def run(self, prompt: str, thinking: bool = False) -> str:
        return prompt
""",
    )
    assert scan_override_signature_drift(tmp_path) == []


def test_kwargs_on_the_base_is_exempt(tmp_path: Path):
    _write(
        tmp_path,
        "passthrough.py",
        """
class Base:
    def run(self, prompt: str, **kwargs) -> str:
        return prompt


class First(Base):
    def run(self, prompt: str, thinking: bool = False, **kwargs) -> str:
        return prompt


class Second(Base):
    def run(self, prompt: str, thinking: bool = False, **kwargs) -> str:
        return prompt
""",
    )
    assert scan_override_signature_drift(tmp_path) == []


def test_overload_stubs_are_exempt(tmp_path: Path):
    _write(
        tmp_path,
        "overloads.py",
        """
from typing import overload


class Base:
    def run(self, prompt: str) -> str:
        return prompt


class First(Base):
    @overload
    def run(self, prompt: str, thinking: bool = False) -> str:
        ...

    def run(self, prompt: str) -> str:
        return prompt


class Second(Base):
    @overload
    def run(self, prompt: str, thinking: bool = False) -> str:
        ...

    def run(self, prompt: str) -> str:
        return prompt
""",
    )
    assert scan_override_signature_drift(tmp_path) == []


def test_underscore_placeholder_rename_is_exempt(tmp_path: Path):
    _write(
        tmp_path,
        "hook.py",
        """
class Base:
    def repair(self, _body: dict, _status: int) -> dict:
        return {}


class First(Base):
    def repair(self, body: dict, status: int) -> dict:
        return body


class Second(Base):
    def repair(self, body: dict, status: int) -> dict:
        return body
""",
    )
    assert scan_override_signature_drift(tmp_path) == []


def test_dunder_overrides_are_exempt(tmp_path: Path):
    _write(
        tmp_path,
        "dunders.py",
        """
class Base:
    def __init__(self, name: str) -> None:
        self.name = name


class First(Base):
    def __init__(self, name: str, retries: int = 3) -> None:
        self.name = name


class Second(Base):
    def __init__(self, name: str, retries: int = 3) -> None:
        self.name = name
""",
    )
    assert scan_override_signature_drift(tmp_path) == []


def test_ambiguous_base_name_is_skipped(tmp_path: Path):
    """Two classes share a name, so no subclass parameter can be attributed to one of them."""
    _write(
        tmp_path,
        "one.py",
        """
class Base:
    def run(self, prompt: str) -> str:
        return prompt
""",
    )
    _write(
        tmp_path,
        "two.py",
        """
class Base:
    def run(self, prompt: str, thinking: bool = False) -> str:
        return prompt


class First(Base):
    def run(self, prompt: str, thinking: bool = False) -> str:
        return prompt


class Second(Base):
    def run(self, prompt: str, thinking: bool = False) -> str:
        return prompt
""",
    )
    assert scan_override_signature_drift(tmp_path) == []


def test_a_matching_override_is_clean(tmp_path: Path):
    _write(
        tmp_path,
        "clean.py",
        """
class Base:
    def run(self, prompt: str, thinking: bool = False) -> str:
        return prompt


class First(Base):
    def run(self, prompt: str, thinking: bool = False) -> str:
        return prompt


class Second(Base):
    def run(self, prompt: str, thinking: bool = False) -> str:
        return prompt
""",
    )
    assert scan_override_signature_drift(tmp_path) == []


def test_grandparent_base_is_reached_through_the_middle_class(tmp_path: Path):
    _write(
        tmp_path,
        "deep.py",
        """
class Root:
    def run(self, prompt: str) -> str:
        return prompt


class Middle(Root):
    pass


class First(Middle):
    def run(self, prompt: str, thinking: bool = False) -> str:
        return prompt


class Second(Middle):
    def run(self, prompt: str, thinking: bool = False) -> str:
        return prompt
""",
    )
    findings = scan_override_signature_drift(tmp_path)
    assert len(findings) == 1
    assert "Root.run" in findings[0].detail

"""Scanner tests for settings_container_field_needs_nodecode: the 12-H1 shape, and each exclusion."""
from __future__ import annotations

from pathlib import Path

from pyutilz.dev.code_audit import scan_settings_container_field_needs_nodecode

from ._helpers import _write

_HEAD = """
from typing import Annotated, Optional
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
"""


def _scan(tmp_path: Path, body: str) -> list:
    _write(tmp_path, "config.py", _HEAD + body)
    return scan_settings_container_field_needs_nodecode(tmp_path)


def test_the_12_h1_precedent_is_flagged(tmp_path: Path):
    """A set field with a comma-splitting before-validator and no NoDecode: the validator never sees the env."""
    findings = _scan(tmp_path, """
class Settings(BaseSettings):
    supported_languages: set[str] = {"en"}

    @field_validator("supported_languages", mode="before")
    @classmethod
    def _split(cls, v):
        return set(v.split(",")) if isinstance(v, str) else v
""")
    assert [(f.check, f.severity) for f in findings] == [("settings_container_field_needs_nodecode", "P1")]
    assert "Settings.supported_languages" in findings[0].detail
    assert findings[0].snippet.startswith("supported_languages: set[str]")


def test_nodecode_clears_it(tmp_path: Path):
    assert _scan(tmp_path, """
class Settings(BaseSettings):
    origins: Annotated[list[str], NoDecode] = []

    @field_validator("origins", mode="before")
    @classmethod
    def _split(cls, v):
        return v.split(",") if isinstance(v, str) else v
""") == []


def test_an_optional_or_union_container_is_still_a_container(tmp_path: Path):
    findings = _scan(tmp_path, """
class Settings(BaseSettings):
    hosts: list[str] | None = None
    proxies: Optional[list[str]] = None

    @field_validator("hosts", "proxies", mode="before")
    @classmethod
    def _split(cls, v):
        return v.split(",") if isinstance(v, str) else v
""")
    assert sorted(f.line for f in findings) == [6, 7]


def test_an_indirect_settings_subclass_is_scanned(tmp_path: Path):
    findings = _scan(tmp_path, """
class Base(BaseSettings):
    pass

class Settings(Base):
    hosts: list[str] = []

    @field_validator("hosts", mode="before")
    @classmethod
    def _split(cls, v):
        return v.split(",") if isinstance(v, str) else v
""")
    assert len(findings) == 1


def test_what_is_not_flagged(tmp_path: Path):
    """An after-validator, a scalar field, an excluded field, decoding switched off, and a plain BaseModel."""
    assert _scan(tmp_path, """
class AfterOnly(BaseSettings):
    hosts: list[str] = []

    @field_validator("hosts", mode="after")
    @classmethod
    def _check(cls, v):
        return v

class Scalar(BaseSettings):
    level: str = "INFO"

    @field_validator("level", mode="before")
    @classmethod
    def _upper(cls, v):
        return v.upper()

class Excluded(BaseSettings):
    cache: dict[str, int] = Field(default_factory=dict, exclude=True)

    @field_validator("cache", mode="before")
    @classmethod
    def _load(cls, v):
        return v

class NoDecoding(BaseSettings):
    model_config = SettingsConfigDict(enable_decoding=False)
    hosts: list[str] = []

    @field_validator("hosts", mode="before")
    @classmethod
    def _split(cls, v):
        return v.split(",") if isinstance(v, str) else v

class Payload(BaseModel):
    tags: list[str] = []

    @field_validator("tags", mode="before")
    @classmethod
    def _split(cls, v):
        return v.split(",") if isinstance(v, str) else v
""") == []


def test_a_v1_pre_validator_counts_as_before(tmp_path: Path):
    findings = _scan(tmp_path, """
from pydantic import validator

class Settings(BaseSettings):
    hosts: list[str] = []

    @validator("hosts", pre=True)
    def _split(cls, v):
        return v.split(",") if isinstance(v, str) else v
""")
    assert len(findings) == 1

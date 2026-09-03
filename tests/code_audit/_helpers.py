"""Fixtures and snippet builders shared by more than one tests/code_audit module.

Split out of the former single-file ``tests/test_code_audit.py`` so each scanner family
keeps its own module: everything below is used from at least two of them.
"""
from __future__ import annotations

from pathlib import Path


def _write(tmp_path: Path, name: str, source: str) -> Path:
    p = tmp_path / name
    p.write_text(source.lstrip("\n"), encoding="utf-8")
    return p


# --- field/text agreement: a structured field vs the free text that duplicates it ---------------------


def _temporal_rule():
    """A representative FieldTextRule: the forensic temporal_class pair, cues and anti-cues included."""
    from pyutilz.dev.code_audit import FieldTextRule

    return FieldTextRule(
        name="temporal_class",
        field="temporal_class",
        text_fields=("subject", "object"),
        cues={"antemortem": ("antemortem", "vital", "vitality"), "postmortem": ("postmortem", "putrefaction")},
        anti_cues={"antemortem": ("vital organs",)},
        neutral_values=frozenset({"na", ""}),
        partitions=(frozenset({"antemortem", "perimortem", "agonal"}), frozenset({"postmortem", "artifact"})),
    )

HELPER = '\nclass Stats:\n    def _inc_stat(self, key, delta=1):\n        with self._lock:\n            self.stats[key] += delta\n\n    def use(self):\n        self._inc_stat("pages")\n'


# --- 2026-09-03 audit: code_audit infrastructure invariants ----------------
#
# The class these pin: a scanner could be registered, exported and unit-tested yet never actually
# run (F04, F09, F10, F199, F200, F207, F210). The registry is the single source of truth; every
# assertion below is derived from it rather than from a hand-maintained list.


def _scanner_function(fn):
    """The underlying function of a registered scanner, unwrapping functools.partial."""
    import functools

    while isinstance(fn, functools.partial):
        fn = fn.func
    return fn

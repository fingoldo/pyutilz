"""Scanner tests for domain_boundary, split verbatim out of the former tests/test_code_audit.py."""
from __future__ import annotations

from pathlib import Path

from ._helpers import _write

# --- domain boundary: domain vocabulary inside code declared domain-neutral --------------------


_BOUNDARY_SOURCE = '''
"""A module holding one neutral concept and one domain concept, not yet split."""


class Envelope:
    """Per-assertion lineage: who observed it, where it is written down."""

    source_id: str = ""
    quote: str = ""


def pool(observations):
    """Pool independent observations of one claim."""
    return sum(observations)


def rank_causes_of_death(rows):
    """Rank the autopsy findings by how well they explain the decedent."""
    return sorted(rows)
'''


def _boundary_tree(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir(exist_ok=True)
    _write(tmp_path, "pkg/envelope.py", _BOUNDARY_SOURCE)
    return tmp_path


def test_domain_vocabulary_leak_is_silent_until_configured(tmp_path: Path):
    from pyutilz.dev.code_audit import BoundarySymbol, scan_domain_vocabulary_leak

    root = _boundary_tree(tmp_path)
    assert scan_domain_vocabulary_leak(root) == []
    assert scan_domain_vocabulary_leak(root, boundary=[BoundarySymbol("pkg/envelope.py", "pool")], vocabulary=[]) == []
    assert scan_domain_vocabulary_leak(root, boundary=[], vocabulary=["autopsy"]) == []


def test_domain_vocabulary_leak_passes_a_clean_boundary_symbol(tmp_path: Path):
    from pyutilz.dev.code_audit import BoundarySymbol, scan_domain_vocabulary_leak

    root = _boundary_tree(tmp_path)
    findings = scan_domain_vocabulary_leak(
        root,
        boundary=[BoundarySymbol("pkg/envelope.py", "Envelope"), BoundarySymbol("pkg/envelope.py", "pool")],
        vocabulary=["autopsy", "decedent", "postmortem"],
    )
    assert findings == []


def test_domain_vocabulary_leak_flags_a_term_in_a_docstring_of_a_boundary_symbol(tmp_path: Path):
    """The leak that matters most is prose: a docstring is where a reader learns what the code is about."""
    from pyutilz.dev.code_audit import BoundarySymbol, scan_domain_vocabulary_leak

    root = _boundary_tree(tmp_path)
    findings = scan_domain_vocabulary_leak(
        root,
        boundary=[BoundarySymbol("pkg/envelope.py", "rank_causes_of_death", note="ranking is neutral")],
        vocabulary=["autopsy", "decedent"],
    )
    assert {f.check for f in findings} == {"domain_vocabulary_leak"}
    assert sorted(f.detail.split("domain term ")[1].split(" ")[0] for f in findings) == ["'autopsy'", "'decedent'"]
    assert "ranking is neutral" in findings[0].detail
    # Two leaks in ONE symbol must stay distinguishable after a ratchet truncates the detail, or the
    # second term would be silently absorbed by the first one's baseline entry.
    assert len({f.detail[:110] for f in findings}) == 2


def test_domain_vocabulary_leak_ignores_the_domain_outside_the_boundary(tmp_path: Path):
    """A term in a sibling symbol is not a leak: the boundary is the claim, not the file."""
    from pyutilz.dev.code_audit import BoundarySymbol, scan_domain_vocabulary_leak

    root = _boundary_tree(tmp_path)
    findings = scan_domain_vocabulary_leak(
        root,
        boundary=[BoundarySymbol("pkg/envelope.py", "pool")],
        vocabulary=["autopsy", "decedent"],
    )
    assert findings == []


def test_domain_vocabulary_leak_honours_an_allowed_term(tmp_path: Path):
    from pyutilz.dev.code_audit import BoundarySymbol, scan_domain_vocabulary_leak

    root = _boundary_tree(tmp_path)
    kwargs = dict(boundary=[BoundarySymbol("pkg/envelope.py", "rank_causes_of_death")], vocabulary=["autopsy", "decedent"])
    assert len(scan_domain_vocabulary_leak(root, **kwargs)) == 2
    assert len(scan_domain_vocabulary_leak(root, allowed=["decedent"], **kwargs)) == 1


def test_domain_vocabulary_leak_matches_on_word_boundaries_not_substrings(tmp_path: Path):
    """`death` must not fire on `deathless` -- a substring rule would make the vocabulary unusable."""
    from pyutilz.dev.code_audit import BoundarySymbol, scan_domain_vocabulary_leak

    _write(tmp_path, "sub.py", '''
def pool(x):
    """A deathless abstraction."""
    return x
''')
    findings = scan_domain_vocabulary_leak(tmp_path, boundary=[BoundarySymbol("sub.py", "pool")], vocabulary=["death"])
    assert findings == []


def test_domain_boundary_reports_a_stale_manifest_rather_than_passing_by_vacuity(tmp_path: Path):
    """A renamed symbol must fail loudly: a boundary that names nothing passes for the wrong reason."""
    from pyutilz.dev.code_audit import BoundarySymbol, scan_domain_vocabulary_leak

    root = _boundary_tree(tmp_path)
    findings = scan_domain_vocabulary_leak(
        root,
        boundary=[BoundarySymbol("pkg/envelope.py", "pool_renamed_away"), BoundarySymbol("pkg/gone.py", "anything")],
        vocabulary=["autopsy"],
    )
    assert [f.check for f in findings] == ["boundary_symbol_missing", "boundary_symbol_missing"]
    assert findings and all(f.severity == "P1" for f in findings)


def test_domain_vocabulary_leak_reaches_a_method_of_a_class(tmp_path: Path):
    from pyutilz.dev.code_audit import BoundarySymbol, scan_domain_vocabulary_leak

    _write(tmp_path, "meth.py", '''
class Store:
    def neutral(self, x):
        return x

    def pool(self, rows):
        """Pool over autopsy series."""
        return rows
''')
    boundary = [BoundarySymbol("meth.py", "Store.neutral"), BoundarySymbol("meth.py", "Store.pool")]
    findings = scan_domain_vocabulary_leak(tmp_path, boundary=boundary, vocabulary=["autopsy"])
    assert [(f.check, f.line) for f in findings] == [("domain_vocabulary_leak", 6)]


def test_getattr_unknown_attribute_catches_a_printer_reading_a_field_that_never_existed(tmp_path):
    """The regression this rule was written for, reduced to its shape.

    A demonstration script printed two headline panels as empty because it asked one dataclass for `steps`
    and another for `lines` - neither had ever existed - and `getattr(obj, name, None) or []` swallowed both.
    The work behind the panels was computed and paid for on every run, and strategy was argued from a blank.
    """
    from pyutilz.dev.code_audit import scan_getattr_unknown_attribute

    (tmp_path / "model.py").write_text(
        "from dataclasses import dataclass, field\n"
        "\n"
        "@dataclass\n"
        "class Sheet:\n"
        "    ask: list = field(default_factory=list)\n"
        "    notes: list = field(default_factory=list)\n",
        encoding="utf-8",
    )
    (tmp_path / "printer.py").write_text(
        "def show(sheet):\n"
        "    for row in getattr(sheet, 'steps', None) or []:\n"
        "        print(row)\n"
        "    for row in getattr(sheet, 'ask', None) or []:\n"
        "        print(row)\n",
        encoding="utf-8",
    )

    findings = scan_getattr_unknown_attribute(tmp_path)
    assert [(f.file, f.line) for f in findings] == [("printer.py", 2)]
    assert "'steps'" in findings[0].detail
    assert findings[0].severity == "P1"


def test_getattr_unknown_attribute_does_not_fire_on_names_the_tree_uses_as_attributes(tmp_path):
    """A name assigned as an attribute anywhere is evidence it exists - including on objects we do not define.

    `threading.local()` and plain namespace objects gain their attributes by assignment and by nothing else,
    so a rule that only read class bodies would report every such lookup as a miss.
    """
    from pyutilz.dev.code_audit import scan_getattr_unknown_attribute

    (tmp_path / "state.py").write_text(
        "import threading\n"
        "\n"
        "CACHE_DIR = '/tmp'\n"
        "_local = threading.local()\n"
        "\n"
        "def open_db():\n"
        "    _local.handle = object()\n"
        "\n"
        "def read(module):\n"
        "    a = getattr(_local, 'handle', None)\n"
        "    b = getattr(module, 'CACHE_DIR', None)\n"
        "    return a, b\n",
        encoding="utf-8",
    )
    assert scan_getattr_unknown_attribute(tmp_path) == []


def test_getattr_unknown_attribute_does_not_fire_on_module_level_def_or_import_bindings(tmp_path):
    """Real false-positive pattern found dogfooding this scanner on pyutilz's own source
    (2026-08-04, ``performance/kernel_tuning/cache/cache_base.py``): a common facade-patchability
    idiom is ``getattr(some_module, "func_name", func_name)`` -- looking a name up on a LIVE
    module object (so a test's ``monkeypatch.setattr(module, "func_name", ...)`` is honored) with
    the in-tree function/import as the fallback. The module-level-bindings widening this scanner
    already documents ("since `getattr(some_module, 'NAME', default)` is a legitimate pattern")
    only walked module-level `Assign`/`AnnAssign` though, missing `def`/`class`/`import` bindings
    entirely -- both a module-level function AND a module-level `from x import y` name must count
    as known."""
    from pyutilz.dev.code_audit import scan_getattr_unknown_attribute

    (tmp_path / "facade.py").write_text(
        "import sys\n"
        "from math import sqrt\n"
        "\n"
        "def _probe() -> int:\n"
        "    return 1\n"
        "\n"
        "class Widget:\n"
        "    pass\n"
        "\n"
        "def use_facade():\n"
        "    _facade = sys.modules[__name__]\n"
        "    probe = getattr(_facade, '_probe', _probe)\n"
        "    root = getattr(_facade, 'sqrt', sqrt)\n"
        "    widget_cls = getattr(_facade, 'Widget', Widget)\n"
        "    return probe(), root(4), widget_cls()\n",
        encoding="utf-8",
    )
    assert scan_getattr_unknown_attribute(tmp_path) == []


def test_getattr_unknown_attribute_ignores_the_two_argument_form(tmp_path):
    """A two-argument getattr raises on a miss, which is loud. The default is what makes the mistake silent."""
    from pyutilz.dev.code_audit import scan_getattr_unknown_attribute

    (tmp_path / "loud.py").write_text("def f(o):\n    return getattr(o, 'nowhere')\n", encoding="utf-8")
    assert scan_getattr_unknown_attribute(tmp_path) == []


def test_getattr_unknown_attribute_accepts_out_of_tree_names(tmp_path):
    """`extra_known` is how a project states that an attribute belongs to a class it does not define."""
    from pyutilz.dev.code_audit import scan_getattr_unknown_attribute

    (tmp_path / "client.py").write_text("def f(provider):\n    return getattr(provider, 'last_generation_id', '')\n", encoding="utf-8")
    assert len(scan_getattr_unknown_attribute(tmp_path)) == 1
    assert scan_getattr_unknown_attribute(tmp_path, extra_known=frozenset({"last_generation_id"})) == []


def test_getattr_literal_on_known_dataclass_catches_a_field_that_belongs_to_a_different_class(tmp_path):
    """Sharper than scan_getattr_unknown_attribute: `steps` is a real field, just not on `Sheet`.

    The union-based rule would miss this because `steps` IS an attribute of something in the tree
    (`Plan`); only per-function local type-tracking catches that the object actually being read
    from is the wrong class for that name.
    """
    from pyutilz.dev.code_audit import scan_getattr_literal_on_known_dataclass

    (tmp_path / "model.py").write_text(
        "from dataclasses import dataclass, field\n"
        "\n"
        "@dataclass\n"
        "class Sheet:\n"
        "    ask: list = field(default_factory=list)\n"
        "    notes: list = field(default_factory=list)\n"
        "\n"
        "@dataclass\n"
        "class Plan:\n"
        "    steps: list = field(default_factory=list)\n",
        encoding="utf-8",
    )
    (tmp_path / "printer.py").write_text(
        "from model import Sheet\n" "\n" "def show():\n" "    sheet = Sheet()\n" "    for row in getattr(sheet, 'steps', None) or []:\n" "        print(row)\n",
        encoding="utf-8",
    )

    findings = scan_getattr_literal_on_known_dataclass(tmp_path)
    assert [(f.file, f.line) for f in findings] == [("printer.py", 5)]
    assert "'steps'" in findings[0].detail and "Sheet" in findings[0].detail


def test_getattr_literal_on_known_dataclass_does_not_fire_on_its_own_real_field(tmp_path):
    from pyutilz.dev.code_audit import scan_getattr_literal_on_known_dataclass

    (tmp_path / "model.py").write_text(
        "from dataclasses import dataclass, field\n" "\n" "@dataclass\n" "class Sheet:\n" "    ask: list = field(default_factory=list)\n",
        encoding="utf-8",
    )
    (tmp_path / "printer.py").write_text(
        "from model import Sheet\n" "\n" "def show():\n" "    sheet = Sheet()\n" "    return getattr(sheet, 'ask', None)\n",
        encoding="utf-8",
    )
    assert scan_getattr_literal_on_known_dataclass(tmp_path) == []


def test_getattr_literal_on_known_dataclass_does_not_fire_when_the_type_cannot_be_inferred(tmp_path):
    """Duck-typing across an intentional boundary is the escape hatch, not a false positive to fix."""
    from pyutilz.dev.code_audit import scan_getattr_literal_on_known_dataclass

    (tmp_path / "model.py").write_text(
        "from dataclasses import dataclass, field\n" "\n" "@dataclass\n" "class Sheet:\n" "    ask: list = field(default_factory=list)\n",
        encoding="utf-8",
    )
    (tmp_path / "printer.py").write_text(
        "def show(sheet):\n" "    return getattr(sheet, 'steps', None)\n",
        encoding="utf-8",
    )
    assert scan_getattr_literal_on_known_dataclass(tmp_path) == []


def test_getattr_literal_on_known_dataclass_infers_type_from_a_parameter_annotation(tmp_path):
    from pyutilz.dev.code_audit import scan_getattr_literal_on_known_dataclass

    (tmp_path / "model.py").write_text(
        "from dataclasses import dataclass, field\n" "\n" "@dataclass\n" "class Sheet:\n" "    ask: list = field(default_factory=list)\n",
        encoding="utf-8",
    )
    (tmp_path / "printer.py").write_text(
        "from model import Sheet\n" "\n" "def show(sheet: Sheet):\n" "    return getattr(sheet, 'steps', None)\n",
        encoding="utf-8",
    )
    findings = scan_getattr_literal_on_known_dataclass(tmp_path)
    assert [(f.file, f.line) for f in findings] == [("printer.py", 4)]

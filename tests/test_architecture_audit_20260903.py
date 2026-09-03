"""Behavioural regression tests for the 2026-09-03 architecture/design audit fixes.

Each test pins one contract that was demonstrably broken before the fix: a subpackage name that did
not resolve as an attribute, an exception the domain root did not catch, two sibling providers
disagreeing about what position in a pricing tuple means, a "is this a test file" predicate that six
scanners answered six different ways, or a hand-maintained list that had drifted from the registry.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import pytest

# --- F04: every subpackage's __all__ is attribute-reachable --------------------------------------

_SUBPACKAGES = ("data", "dev", "text", "database", "core", "cloud", "stats", "performance", "system", "llm")


@pytest.mark.parametrize("subpackage", _SUBPACKAGES)
def test_every_subpackage_all_entry_is_attribute_reachable(subpackage):
    """`import pyutilz.X as p; p.name` must work for every name in `pyutilz.X.__all__`.

    Six subpackages promised submodule names their package had no attribute for, so the three
    spellings of one import disagreed: `from pyutilz.data import pandaslib` and `import *` bound it,
    `pyutilz.data.pandaslib` raised AttributeError with no import-time warning.
    """
    module = importlib.import_module(f"pyutilz.{subpackage}")
    missing = sorted(name for name in getattr(module, "__all__", ()) if not hasattr(module, name))
    assert not missing, f"pyutilz.{subpackage}.__all__ names unreachable as attributes: {missing}"


@pytest.mark.parametrize("subpackage", ("data", "dev", "text", "database", "core"))
def test_subpackage_getattr_still_raises_for_an_unknown_name(subpackage):
    """The PEP 562 hook must not turn every typo into an import attempt."""
    module = importlib.import_module(f"pyutilz.{subpackage}")
    with pytest.raises(AttributeError):
        getattr(module, "NoSuchSubmoduleName")


def test_data_pandaslib_resolves_as_an_attribute():
    import pyutilz.data

    assert pyutilz.data.pandaslib.__name__ == "pyutilz.data.pandaslib"


# --- F05: ClaudeCodeToolUseError is rooted at the domain root ------------------------------------


def test_claude_code_tool_use_error_is_caught_by_the_domain_root():
    from pyutilz.llm.exceptions import ClaudeCodeToolUseError, LLMProviderError

    try:
        raise ClaudeCodeToolUseError("tool-use block escaped")
    except LLMProviderError:
        pass
    else:  # pragma: no cover - the raise above always fires
        pytest.fail("ClaudeCodeToolUseError escaped `except LLMProviderError`")


def test_claude_code_tool_use_error_keeps_its_runtime_error_identity():
    """Still a RuntimeError and still NOT an OSError, so the transient-retry arm cannot swallow it."""
    from pyutilz.llm.exceptions import ClaudeCodeToolUseError

    assert issubclass(ClaudeCodeToolUseError, RuntimeError)
    assert not issubclass(ClaudeCodeToolUseError, OSError)


def test_claude_code_tool_use_error_is_importable_from_the_domain_facade():
    import pyutilz.llm as llm_facade
    from pyutilz.llm.claude_code_provider import ClaudeCodeToolUseError as ProviderClaudeCodeToolUseError

    assert llm_facade.ClaudeCodeToolUseError is ProviderClaudeCodeToolUseError
    assert "ClaudeCodeToolUseError" in llm_facade.__all__


@pytest.mark.parametrize(
    "package_name, root_name",
    [("pyutilz.llm", "LLMProviderError"), ("pyutilz.database", "DatabaseError"), ("pyutilz.web", "WebError")],
)
def test_no_exception_anywhere_in_a_domain_escapes_its_root(package_name, root_name):
    """The widened guard: walk EVERY module in the domain, not just its `exceptions` module.

    The previous guard iterated `vars(exceptions)` only, so an exception defined in a provider
    module was structurally invisible to it - which is exactly how ClaudeCodeToolUseError sat
    outside the hierarchy undetected.
    """
    package = importlib.import_module(package_name)
    root = getattr(importlib.import_module(package_name + ".exceptions"), root_name)

    escapes = []
    imported = 0
    for info in pkgutil.walk_packages(package.__path__, prefix=package_name + "."):
        try:
            module = importlib.import_module(info.name)
        except ImportError:  # an optional third-party dependency is not installed here
            continue
        imported += 1
        for name, obj in vars(module).items():
            if not (isinstance(obj, type) and issubclass(obj, BaseException)):
                continue
            if getattr(obj, "__module__", "") != info.name:
                continue  # defined elsewhere and merely imported here
            if obj is root or issubclass(obj, root):
                continue
            escapes.append(f"{info.name}.{name}")
    # A domain whose modules all failed to import would report "no escapes" while checking nothing.
    assert imported >= 3, f"only {imported} module(s) of {package_name} could be imported"
    assert not escapes, f"exception(s) escaping `except {root_name}`: {sorted(escapes)}"


# --- F06: one test-file predicate -----------------------------------------------------------------


def test_is_test_file_accepts_every_convention_the_scanners_used_to_disagree_about():
    from pyutilz.dev.code_audit._base import is_test_file

    root = Path("/repo")
    assert is_test_file(root / "tests" / "test_a.py", root)
    assert is_test_file(root / "pkg" / "a_test.py", root)
    assert is_test_file(root / "tests" / "check_pricing.py", root)  # no prefix, but under tests/
    assert is_test_file(root / "test" / "helpers.py", root)
    assert not is_test_file(root / "src" / "pkg" / "module.py", root)


def test_is_test_file_directory_check_is_relative_to_the_scan_root():
    """`Path.parts` is absolute: a checkout living under a directory named `tests` must not make
    every production module in it read as test code."""
    from pyutilz.dev.code_audit._base import is_test_file

    # Built from parts rather than written as a literal: a checkout path spelled out in a test is
    # a path that exists on exactly one machine.
    root = Path("checkouts").joinpath("tests", "myproject").resolve()
    assert not is_test_file(root / "src" / "module.py", root)


def test_is_test_file_extra_globs_widen_it_for_a_custom_layout():
    from pyutilz.dev.code_audit._base import is_test_file

    root = Path("/repo")
    assert not is_test_file(root / "src" / "check_pricing.py", root)
    assert is_test_file(root / "src" / "check_pricing.py", root, ("check_*.py",))


def test_no_scanner_module_defines_its_own_test_file_predicate():
    """The whole point of the shared helper: thirteen private re-implementations, six semantics."""
    import pyutilz.dev.code_audit as facade

    package_dir = Path(facade.__file__).parent
    offenders = []
    for py in sorted(package_dir.glob("*.py")):
        text = py.read_text(encoding="utf-8", errors="replace")
        for marker in ("def _is_test_file", "def _looks_like_a_test_file", "def _is_test_path"):
            if marker in text:
                offenders.append(f"{py.name}:{marker}")
    assert not offenders, f"private test-file predicates still defined: {offenders}"


_TESTS_DIR_FIXTURE = '''
import inspect

import pytest

import mod


def test_pricing():
    path = "D:/Machine Learning/data.csv"
    try:
        value = mod.compute(path)
    except Exception:
        pytest.skip("backend missing")
    assert True
    src = inspect.getsource(mod.compute)
    assert "AT TIME ZONE 'utc'" in src
'''


@pytest.mark.parametrize(
    "scanner_name",
    ["hardcoded_absolute_path_in_test", "source_text_assertion", "vacuous_assertion", "except_skip_masks_call_under_test"],
)
def test_test_focused_scanners_agree_on_a_tests_dir_file_without_a_test_prefix(tmp_path, scanner_name):
    """A file at `tests/check_pricing.py` is test code for EVERY test-focused scanner, or for none.

    Before the shared predicate, half this catalogue skipped such a file entirely and reported
    nothing, indistinguishable from a clean result.
    """
    from pyutilz.dev.code_audit import run_all

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "check_pricing.py").write_text(_TESTS_DIR_FIXTURE, encoding="utf-8")
    findings = run_all(tmp_path, checks=[scanner_name])
    assert findings, f"{scanner_name} reported nothing for a tests/ file without a test_ prefix"


# --- F07: the catalogue is derived from the registry ---------------------------------------------


def test_describe_scanners_covers_every_registered_check():
    from pyutilz.dev.code_audit import describe_scanners, get_scanners

    assert set(describe_scanners()) == set(get_scanners())


def test_describe_scanners_marks_exactly_the_opt_in_checks():
    from pyutilz.dev.code_audit import OPT_IN_ONLY, describe_scanners

    marked = {name for name, entry in describe_scanners().items() if entry["opt_in"]}
    assert marked == set(OPT_IN_ONLY)


def test_every_described_check_carries_a_summary():
    from pyutilz.dev.code_audit import describe_scanners

    undescribed = sorted(name for name, entry in describe_scanners().items() if not entry["summary"])
    assert not undescribed, f"registered check(s) with no summary line: {undescribed}"


def test_cli_list_checks_prints_the_catalogue_and_marks_opt_ins(capsys):
    from pyutilz.dev.code_audit import OPT_IN_ONLY, get_scanners, main

    assert main(["--list-checks"]) == 0
    out = capsys.readouterr().out
    for name in get_scanners():
        assert name in out, f"{name} missing from --list-checks output"
    for name in OPT_IN_ONLY:
        assert f"{name} (OPT-IN)" in out


# --- F09: the submodule sweep is computed, not hand-maintained ------------------------------------


def test_no_scanner_submodule_leaks_onto_the_facade_attribute_surface():
    import types

    import pyutilz.dev.code_audit as facade

    leaked = sorted(name for name, value in vars(facade).items() if isinstance(value, types.ModuleType) and value.__name__.startswith(facade.__name__ + "."))
    assert not leaked, f"submodule(s) still bound on the facade: {leaked}"


def test_the_deliberate_stdlib_reexports_survive_the_sweep():
    import pyutilz.dev.code_audit as facade

    assert facade.ast.__name__ == "ast"
    assert facade.json.__name__ == "json"
    assert facade.sys.__name__ == "sys"


def test_a_submodule_is_still_importable_on_demand():
    from pyutilz.dev.code_audit import closures

    assert closures.__name__ == "pyutilz.dev.code_audit.closures"


# --- F08 / F12: one pricing contract across providers ---------------------------------------------


def test_pricing_named_fields_survive_a_provider_written_against_the_wrong_position():
    from pyutilz.llm.openai_compat import Pricing

    pricing = Pricing(input=1.0, output=8.0, cache_hit=0.1)
    assert pricing.input == 1.0
    assert pricing.output == 8.0
    assert pricing.cache_hit == 0.1


def test_pricing_cache_hit_defaults_to_none_meaning_no_published_rate():
    from pyutilz.llm.openai_compat import Pricing

    assert Pricing(1.0, 8.0).cache_hit is None


@pytest.mark.parametrize(
    "module_name, class_name, model",
    [
        ("pyutilz.llm.deepseek_provider", "DeepSeekProvider", "deepseek-v4-flash"),
        ("pyutilz.llm.xai_provider", "XAIProvider", "grok-4-fast"),
        ("pyutilz.llm.openai_provider", "OpenAIProvider", "gpt-5-mini"),
    ],
)
def test_sibling_providers_agree_on_what_resolve_pricing_returns(module_name, class_name, model):
    """The divergence that made this a finding: the same private method name returned a 2-tuple in
    one provider and a 3-tuple in its neighbour, so the accessors indexed [1] and [2] for the same
    quantity and a copied accessor would silently report the cache-hit rate as the output rate."""
    from pyutilz.llm.openai_compat import Pricing

    provider_class = getattr(importlib.import_module(module_name), class_name)
    pricing = provider_class._resolve_pricing(object.__new__(provider_class), model)
    assert isinstance(pricing, Pricing)
    assert pricing.output == provider_class._output_cost_per_1m(object.__new__(provider_class), model)
    assert pricing.input == provider_class._input_cost_per_1m(object.__new__(provider_class), model)


def test_deepseek_output_rate_is_not_the_cache_hit_rate():
    """DeepSeek's own table is stored (input, cache_hit, output); the reorder into Pricing's named
    fields is what keeps the output accessor off the cache-hit column."""
    from pyutilz.llm.deepseek_provider import DeepSeekProvider, _PRICING

    provider = object.__new__(DeepSeekProvider)
    in_cost, cache_hit, out_cost = _PRICING["deepseek-v4-flash"]
    pricing = DeepSeekProvider._resolve_pricing(provider, "deepseek-v4-flash")
    assert (pricing.input, pricing.output, pricing.cache_hit) == (in_cost, out_cost, cache_hit)


def test_base_cache_hit_falls_back_to_the_input_rate_when_no_rate_is_published():
    from pyutilz.llm.openai_compat import OpenAICompatibleProvider, Pricing

    class _Stub(OpenAICompatibleProvider):
        def _input_cost_per_1m(self, model):
            return 3.0

        def _output_cost_per_1m(self, model):
            return 9.0

    stub = object.__new__(_Stub)
    assert _Stub._resolve_pricing(stub, "m") == Pricing(3.0, 9.0, None)
    assert _Stub._cache_hit_cost_per_1m(stub, "m") == 3.0


def test_openrouter_prices_cache_hits_from_the_catalogue_cached_input_rate(monkeypatch):
    """OpenRouter tracks cache-hit tokens but used to bill every one at the full input rate."""
    from pyutilz.llm.openrouter_provider import _provider as provider_module
    from pyutilz.llm.openrouter_provider._provider import OpenRouterProvider

    monkeypatch.setattr(provider_module, "_per_token_cost_pair", lambda model: (3.0, 15.0))
    monkeypatch.setattr(provider_module, "_cache_read_cost_per_1m_or_none", lambda model: 0.3)

    provider = object.__new__(OpenRouterProvider)
    assert OpenRouterProvider._resolve_pricing(provider, "m").cache_hit == 0.3
    assert OpenRouterProvider._cache_hit_cost_per_1m(provider, "m") == 0.3


def test_openrouter_falls_back_to_the_input_rate_when_the_catalogue_has_no_cached_rate(monkeypatch):
    from pyutilz.llm.openrouter_provider import _provider as provider_module
    from pyutilz.llm.openrouter_provider._provider import OpenRouterProvider

    monkeypatch.setattr(provider_module, "_per_token_cost_pair", lambda model: (3.0, 15.0))
    monkeypatch.setattr(provider_module, "_cache_read_cost_per_1m_or_none", lambda model: None)

    provider = object.__new__(OpenRouterProvider)
    assert OpenRouterProvider._cache_hit_cost_per_1m(provider, "m") == 3.0


def test_cache_read_cost_reads_the_catalogue_field_and_says_none_when_absent(monkeypatch):
    from pyutilz.llm.openrouter_provider import _catalogue

    monkeypatch.setattr(
        _catalogue,
        "_fetch_models_catalogue",
        lambda *a, **k: {
            "priced": {"pricing": {"prompt": "0.000003", "completion": "0.000015", "input_cache_read": "0.0000003"}},
            "unpriced": {"pricing": {"prompt": "0.000003", "completion": "0.000015"}},
            "garbage": {"pricing": {"input_cache_read": "not-a-number"}},
        },
    )
    assert _catalogue._cache_read_cost_per_1m_or_none("priced") == pytest.approx(0.3)
    assert _catalogue._cache_read_cost_per_1m_or_none("unpriced") is None
    assert _catalogue._cache_read_cost_per_1m_or_none("garbage") is None
    assert _catalogue._cache_read_cost_per_1m_or_none("absent") is None


# --- F13: shared private helpers ------------------------------------------------------------------


def test_both_scanners_share_one_definition_of_a_caching_decorator():
    """`@functools.cache` widened in one copy and not the other was the drift this closes."""
    import ast

    from pyutilz.dev.code_audit._base import is_cached
    from pyutilz.dev.code_audit.redundant_test_fit import is_cached as from_redundant_test_fit
    from pyutilz.dev.code_audit.uncached_constant_cost_probe import is_cached as from_probe

    assert from_redundant_test_fit is is_cached
    assert from_probe is is_cached

    tree = ast.parse("import functools\n\n@functools.cache\ndef f():\n    pass\n")
    func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    assert is_cached(func)


def test_both_scanners_share_one_definition_of_a_scope_call():
    import ast

    from pyutilz.dev.code_audit._base import is_locals_or_globals_call
    from pyutilz.dev.code_audit.locals_get import is_locals_or_globals_call as from_locals_get
    from pyutilz.dev.code_audit.locals_globals_output import is_locals_or_globals_call as from_output

    assert from_locals_get is is_locals_or_globals_call
    assert from_output is is_locals_or_globals_call
    assert is_locals_or_globals_call(ast.parse("locals()", mode="eval").body)
    assert not is_locals_or_globals_call(ast.parse("locals(1)", mode="eval").body)


def test_dotted_name_says_nothing_rather_than_a_partial_for_a_non_name_root():
    import ast

    from pyutilz.dev.code_audit._base import dotted_name

    assert dotted_name(ast.parse("a.b.c", mode="eval").body) == "a.b.c"
    assert dotted_name(ast.parse("registry[k].run", mode="eval").body) == ""


def test_no_private_helper_name_is_defined_twice_across_scanner_modules():
    """The collisions that remain must be genuinely one helper, not two rules sharing a name."""
    import re

    import pyutilz.dev.code_audit as facade

    package_dir = Path(facade.__file__).parent
    definitions: dict[str, list[str]] = {}
    pattern = re.compile(r"^def (_[A-Za-z0-9_]+)\(", re.MULTILINE)
    for py in sorted(package_dir.glob("*.py")):
        for name in set(pattern.findall(py.read_text(encoding="utf-8", errors="replace"))):
            definitions.setdefault(name, []).append(py.name)

    collisions = {name: sorted(files) for name, files in definitions.items() if len(files) > 1}
    known = {"_is_cached", "_is_locals_or_globals_call", "_module_level_names", "_accumulates", "_patched_targets", "_ancestor_chain"}
    assert not (known & set(collisions)), f"helper name(s) still defined twice: { {k: collisions[k] for k in known & set(collisions)} }"


# --- F14: the record checker is not a member of the scanner package -------------------------------


def test_field_text_agreement_lives_outside_the_scanner_package():
    import pyutilz.dev.field_text_agreement as moved

    assert moved.__name__ == "pyutilz.dev.field_text_agreement"
    assert moved.check_record.__module__ == "pyutilz.dev.field_text_agreement"


def test_the_old_import_path_still_resolves_to_the_same_objects():
    import pyutilz.dev.code_audit as facade
    import pyutilz.dev.code_audit.field_text_agreement as shim
    import pyutilz.dev.field_text_agreement as moved

    for name in ("check_record", "check_records", "check_all", "cues_in_text", "FieldTextRule", "AGREE"):
        assert getattr(shim, name) is getattr(moved, name)
        assert getattr(facade, name) is getattr(moved, name)


def test_every_module_left_in_the_scanner_package_is_a_source_scanner():
    """The package's contract, now true without an allowlist entry for the record checker."""
    import pyutilz.dev.code_audit as facade

    package_dir = Path(facade.__file__).parent
    non_scanner = []
    infrastructure = {"__init__.py", "__main__.py", "_base.py", "registry.py", "cli.py", "meta_test_utils.py", "field_text_agreement.py"}
    for py in sorted(package_dir.glob("*.py")):
        if py.name in infrastructure:
            continue
        text = py.read_text(encoding="utf-8", errors="replace")
        if "def scan_" not in text:
            non_scanner.append(py.name)
    assert not non_scanner, f"module(s) in the scanner package that define no scanner: {non_scanner}"


# --- F15: typing generics come from `typing` on the 3.8 floor -------------------------------------


def test_no_scanner_module_imports_typing_generics_from_collections_abc():
    """`collections.abc.Iterator[...]` is not subscriptable on the declared 3.8 floor outside an
    annotation, and the pattern's presence legitimised the next runtime subscript."""
    import pyutilz.dev.code_audit as facade

    package_dir = Path(facade.__file__).parent
    offenders = [py.name for py in sorted(package_dir.glob("*.py")) if "from collections.abc import" in py.read_text(encoding="utf-8", errors="replace")]
    assert not offenders, f"module(s) importing typing generics from collections.abc: {offenders}"

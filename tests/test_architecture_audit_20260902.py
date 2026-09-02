"""Behavioural regression tests for the architecture/design audit fixes.

Each test pins one contract that was demonstrably broken before the fix: a public name that did not
resolve, an exception the domain root did not catch, a library default naming a downstream project,
or a mutable registry reachable as public API.
"""

import importlib
import pkgutil
import warnings
from pathlib import Path

import pytest

import pyutilz

# --- package facade completeness -------------------------------------------------------------


def test_every_subpackage_resolves_as_an_attribute_of_pyutilz():
    """`pyutilz.<sub>` and `from pyutilz import <sub>` must agree for EVERY shipped subpackage.

    They used to disagree for exactly two names: `stats` and `performance` were shipped and
    extras-declared but absent from both `__all__` and the lazy `__getattr__` tuple, so attribute
    access raised AttributeError while the import statement worked through Python's submodule
    fallback.
    """
    root = Path(pyutilz.__file__).resolve().parent
    on_disk = {m.name for m in pkgutil.iter_modules([str(root)]) if m.ispkg}

    missing_attr = sorted(name for name in on_disk if not hasattr(pyutilz, name))
    assert not missing_attr, f"subpackage(s) unreachable as pyutilz.<name>: {missing_attr}"

    missing_all = sorted(on_disk - set(pyutilz.__all__))
    assert not missing_all, f"subpackage(s) missing from pyutilz.__all__: {missing_all}"


def test_stats_and_performance_specifically_resolve():
    assert pyutilz.stats.__name__ == "pyutilz.stats"
    assert pyutilz.performance.__name__ == "pyutilz.performance"


def test_performance_binds_its_submodule_like_every_sibling():
    import pyutilz.performance as perf

    assert perf.kernel_tuning.__name__ == "pyutilz.performance.kernel_tuning"
    assert "kernel_tuning" in perf.__all__


# --- llm public surface -----------------------------------------------------------------------


def test_every_registered_provider_class_is_reachable_from_the_llm_facade():
    """The facade used to keep a hand-copied duplicate of the factory's provider table, and the
    duplicate silently omitted OpenAIProvider."""
    import pyutilz.llm as L
    from pyutilz.llm.factory import _PROVIDER_MODULES

    for canonical, (mod_path, cls_name, _key_attr) in _PROVIDER_MODULES.items():
        obj = getattr(L, cls_name)
        # Identity against the registered module, not `obj.__module__`: a provider may itself be a
        # re-export subpackage (openrouter), where the class's defining module is a submodule.
        assert obj is getattr(importlib.import_module(mod_path), cls_name), canonical
        assert cls_name in L.__all__, f"{cls_name} reachable but missing from pyutilz.llm.__all__"


def test_openai_provider_specifically_resolves():
    import pyutilz.llm as L

    assert L.OpenAIProvider.__name__ == "OpenAIProvider"


def test_every_llm_exception_type_is_reachable_from_the_facade():
    import pyutilz.llm as L
    from pyutilz.llm import exceptions

    declared = [name for name, obj in vars(exceptions).items() if isinstance(obj, type) and issubclass(obj, BaseException)]
    for name in declared:
        assert getattr(L, name) is getattr(exceptions, name), name
        assert name in L.__all__, f"{name} missing from pyutilz.llm.__all__"


def test_unknown_provider_attribute_still_raises_attribute_error():
    """The provider fallback must not turn every *Provider-suffixed typo into an import attempt."""
    import pyutilz.llm as L

    with pytest.raises(AttributeError):
        L.NoSuchProvider


# --- exception hierarchies ---------------------------------------------------------------------


def test_llm_provider_error_catches_every_llm_exception():
    from pyutilz.llm import exceptions

    root = exceptions.LLMProviderError
    for name, obj in vars(exceptions).items():
        if isinstance(obj, type) and issubclass(obj, BaseException) and obj is not root:
            assert issubclass(obj, root), f"{name} escapes `except LLMProviderError`"


@pytest.mark.parametrize("exc_name", ["JSONParsingError", "LLMTruncationError"])
def test_dual_base_keeps_value_error_call_sites_working(exc_name):
    from pyutilz.llm import exceptions

    exc = getattr(exceptions, exc_name)
    assert issubclass(exc, ValueError)
    assert issubclass(exc, exceptions.LLMProviderError)


def test_truncation_error_is_caught_by_the_domain_root_at_runtime():
    from pyutilz.llm.exceptions import LLMProviderError, LLMTruncationError

    try:
        raise LLMTruncationError("hit the cap", finish_reason="length", partial_text="abc")
    except LLMProviderError as exc:
        assert exc.partial_text == "abc"
    else:  # pragma: no cover - the raise above always fires
        pytest.fail("LLMTruncationError escaped `except LLMProviderError`")


def test_database_domain_has_a_catchable_root():
    from pyutilz.database import exceptions

    for name, obj in vars(exceptions).items():
        if isinstance(obj, type) and issubclass(obj, BaseException) and obj is not exceptions.DatabaseError:
            assert issubclass(obj, exceptions.DatabaseError), name
    assert issubclass(exceptions.DatabaseConnectionError, RuntimeError)
    assert issubclass(exceptions.SQLValidationError, ValueError)


def test_web_domain_has_a_catchable_root():
    from pyutilz.web import exceptions

    for name, obj in vars(exceptions).items():
        if isinstance(obj, type) and issubclass(obj, BaseException) and obj is not exceptions.WebError:
            assert issubclass(obj, exceptions.WebError), name
    assert issubclass(exceptions.ProxyFetchError, RuntimeError)
    assert issubclass(exceptions.UnsafeURLError, ValueError)
    # url_guard's historic promise: NOT an OSError, so retry-on-transient-network logic skips it.
    assert not issubclass(exceptions.UnsafeURLError, OSError)


# --- typed exceptions on the package facade -----------------------------------------------------


def test_database_exceptions_are_on_the_package_facade():
    import pyutilz.database as d
    from pyutilz.database.exceptions import DatabaseConnectionError, DatabaseError, SQLValidationError

    assert d.DatabaseError is DatabaseError
    assert d.DatabaseConnectionError is DatabaseConnectionError
    assert d.SQLValidationError is SQLValidationError
    for name in ("exceptions", "DatabaseError", "DatabaseConnectionError", "SQLValidationError"):
        assert name in d.__all__


def test_web_exceptions_are_on_the_package_facade():
    import pyutilz.web as w
    from pyutilz.web.exceptions import ProxyConfigurationError, ProxyFetchError, UnsafeURLError, WebError

    assert (w.WebError, w.ProxyConfigurationError, w.ProxyFetchError, w.UnsafeURLError) == (
        WebError,
        ProxyConfigurationError,
        ProxyFetchError,
        UnsafeURLError,
    )
    for name in ("exceptions", "WebError", "ProxyConfigurationError", "ProxyFetchError", "UnsafeURLError"):
        assert name in w.__all__


def test_url_guard_still_exposes_the_relocated_exception():
    """`from pyutilz.web.url_guard import UnsafeURLError` is the historic path and must keep working."""
    from pyutilz.web import url_guard
    from pyutilz.web.exceptions import UnsafeURLError

    assert url_guard.UnsafeURLError is UnsafeURLError
    with pytest.raises(UnsafeURLError):
        url_guard.require_http_url("file:///etc/passwd")


def test_core_and_system_expose_their_inline_exceptions():
    import pyutilz.core as c
    import pyutilz.system as s
    from pyutilz.core.safe_pickle import PickleVerificationError
    from pyutilz.system.resilience import CircuitOpenError

    assert c.PickleVerificationError is PickleVerificationError
    assert "PickleVerificationError" in c.__all__
    assert s.CircuitOpenError is CircuitOpenError
    assert "CircuitOpenError" in s.__all__


# --- layering ------------------------------------------------------------------------------------


def test_token_helpers_live_in_the_llm_domain_and_the_old_path_still_works():
    from pyutilz.core import openai as shim
    from pyutilz.llm import openai_tokens

    assert shim.num_tokens_from_string is openai_tokens.num_tokens_from_string
    assert shim.num_tokens_from_messages is openai_tokens.num_tokens_from_messages
    assert pyutilz.openai.__name__ == "pyutilz.llm.openai_tokens"


def test_core_openai_shim_does_not_import_llm_at_module_level():
    """The shim resolves lazily ON PURPOSE: a top-level re-export would recreate the core -> llm
    edge that stopped `core` being a leaf layer."""
    import ast

    src = Path(pyutilz.__file__).resolve().parent / "core" / "openai.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    for node in tree.body:  # module level only
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("pyutilz.llm"), ast.dump(node)
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("pyutilz.llm"), alias.name


def test_kernel_tuning_discovery_has_no_downstream_project_default():
    """A general-purpose library must not default its scan target to one consumer's package name:
    every other consumer got an ERROR log about an unknown package and an empty registry back."""
    import inspect

    from pyutilz.performance.kernel_tuning import discover_tuners, retune_all

    for fn in (discover_tuners, retune_all):
        param = inspect.signature(fn).parameters["package"]
        assert param.default is inspect.Parameter.empty, f"{fn.__name__}(package=...) still has a default"
        with pytest.raises(TypeError):
            fn()


# --- text.strings star-export hygiene ---------------------------------------------------------


def test_strings_star_import_no_longer_injects_third_party_or_stdlib_modules():
    ns: dict = {}
    exec("from pyutilz.text.strings import *", ns)  # noqa: S102 - that is precisely what is under test
    shadowed = ("pd", "np", "re", "json", "math", "string", "deque", "Counter", "OrderedDict", "unicodedata", "defaultdict")
    leaked = sorted(n for n in shadowed if n in ns)
    assert not leaked, f"star-import still shadows caller names: {leaked}"
    # The package's own API must still arrive.
    assert callable(ns["slugify"])


def test_strings_legacy_reexports_stay_reachable_by_attribute():
    import collections

    import pyutilz.text.strings as S

    assert S.pd.__name__ == "pandas"
    assert S.np.__name__ == "numpy"
    assert S.Counter is collections.Counter
    assert S.re.__name__ == "re"


def test_strings_import_does_not_pull_pandas_eagerly():
    """The facade referenced neither pandas nor numpy, yet imported both at module load."""
    import ast

    src = Path(pyutilz.__file__).resolve().parent / "text" / "strings" / "__init__.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    eager = {alias.name.split(".")[0] for node in tree.body if isinstance(node, ast.Import) for alias in node.names}
    eager |= {(node.module or "").split(".")[0] for node in tree.body if isinstance(node, ast.ImportFrom)}
    assert "pandas" not in eager and "numpy" not in eager


# --- code_audit registry encapsulation ----------------------------------------------------------


def test_scanner_registry_is_not_public_on_the_facade():
    import pyutilz.dev.code_audit as facade

    assert not hasattr(facade, "SCANNERS")
    assert "SCANNERS" not in facade.__all__
    assert facade.get_scanners()  # the read-only accessor is the supported path


def test_get_scanners_returns_a_copy_so_mutation_cannot_corrupt_run_all():
    from pyutilz.dev.code_audit import get_scanners

    first = get_scanners()
    name = next(iter(first))
    first.pop(name)
    assert name in get_scanners(), "mutating the accessor's result leaked into the shared registry"


def test_deprecated_scanners_alias_warns_and_hands_back_a_copy():
    from pyutilz.dev.code_audit import registry

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy = registry.SCANNERS
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert legacy is not registry._SCANNERS
    assert legacy == registry._SCANNERS


def test_registry_module_still_raises_for_unknown_attributes():
    from pyutilz.dev.code_audit import registry

    with pytest.raises(AttributeError):
        registry.NoSuchName


# --- module size ---------------------------------------------------------------------------------


def test_no_module_exceeds_the_project_split_threshold():
    """The repo's own convention: split a module past 1000 LOC into a re-export subpackage."""
    root = Path(pyutilz.__file__).resolve().parent
    oversized = []
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            n = sum(1 for _ in fh)
        if n > 1000:
            oversized.append(f"{path.relative_to(root).as_posix()}: {n}")
    assert not oversized, "module(s) over the 1000-LOC split threshold:\n  " + "\n  ".join(sorted(oversized))


@pytest.mark.parametrize(
    "modname",
    [
        "pyutilz.text.similarity",
        "pyutilz.database.db",
        "pyutilz.data.polarslib",
        "pyutilz.performance.kernel_tuning.cache.cache_class",
        "pyutilz.web.web",
    ],
)
def test_split_modules_still_import_and_expose_a_surface(modname):
    mod = importlib.import_module(modname)
    public = [n for n in dir(mod) if not n.startswith("_")]
    assert public, f"{modname} exposes nothing after the split"

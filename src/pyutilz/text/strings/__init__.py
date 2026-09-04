# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from pyutilz.core.pythonlib import ensure_installed

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import Any, Iterable, List, Optional, Sequence, Union

from pyutilz.core.pythonlib import is_float

# ----------------------------------------------------------------------------------------------------------------------------
# Names this facade historically re-exported but does not itself use: pandas/numpy and a handful
# of stdlib modules. They stay REACHABLE (``pyutilz.text.strings.pd``) for backward compatibility
# but are resolved lazily and are NOT in ``__all__``, because:
#  * star-importing them injected ``re``/``json``/``math``/``string``/``Counter``... into the
#    caller's namespace, silently shadowing same-named local bindings (a caller's orjson shim
#    bound to ``json``, say) with no error;
#  * the eager ``import pandas as pd, numpy as np`` cost ~1.6 s at import time for a pure
#    string-manipulation package that references neither.
# ----------------------------------------------------------------------------------------------------------------------------

# Annotated as a string literal: this module has no from __future__ import annotations, and the
# PEP 604 union would be evaluated at runtime on the oldest supported interpreter.
_LAZY_REEXPORTS: "dict[str, tuple[str, Optional[str]]]" = {
    "pd": ("pandas", None),
    "np": ("numpy", None),
    "string": ("string", None),
    "json": ("json", None),
    "unicodedata": ("unicodedata", None),
    "re": ("re", None),
    "math": ("math", None),
    "OrderedDict": ("collections", "OrderedDict"),
    "defaultdict": ("collections", "defaultdict"),
    "deque": ("collections", "deque"),
    "Counter": ("collections", "Counter"),
}

# ----------------------------------------------------------------------------------------------------------------------------
# Re-export public API from cohesive submodules (subpackage split of the former 1070-LOC strings.py).
# Every name that used to be importable via `from pyutilz.text.strings import X` remains importable here.
# ----------------------------------------------------------------------------------------------------------------------------

from .jsonutils import (
    json_serial,
    sub_elem,
    jsonize_atrtributes,
    remove_json_attributes,
    leave_json_attributes,
    extract_json_attribute,
    remove_json_empty_attributes,
    remove_json_defaults,
    json_pg_dumps,
    get_jsonlist_property,
    get_jsonlist_properties,
)

from .configfiles import read_config_file, write_config_file

from .basics import (
    find_between,
    parse_tokens,
    make_text_from_inner_html_elements,
    underscorize_variable,
    get_hash,
    strip_characters,
    strip_doubled_characters,
    rpad,
    shorten_path,
    slugify,
    camel_case_split,
)

from .webtext import (
    punctuation,
    eos,
    spacy_sent_tokenize,
    remove_videos,
    fix_duplicate_tokens,
    unescape_html,
    fix_html,
    parse_html,
    fix_quotations,
    fix_spaces,
    fix_broken_sentences,
    fix_missed_space_between_sentences,
    merge_punctuation_signs,
    ensure_space_after_comma,
    clean_description,
    get_ascii_emojies,
    get_unicode_emojies,
    sentencize_text,
    suffixize,
)

from .textentropy import (
    tokenize_text,
    tokenize_source,
    tokenize_to_chars,
    tokenize_to_words,
    get_entropy_stats,
    entropy,
    entropy_rate,
    compute_entropy_stats,
    naive_entropy_rate,
    stringify_dict,
)

# ----------------------------------------------------------------------------------------------------------------------------
# Lazily-initialised module-level globals.
#
# These are stateful caches mutated (via `global`) inside webtext.py. A plain `from .webtext import
# inflect_engine, nlp, ...` would take a ONE-TIME SNAPSHOT (all None) at package-import time --
# webtext.py's own functions later mutate ITS OWN module-level globals via `global nlp` etc., which
# never updates a separately-bound copy here. PEP 562 module-level __getattr__ resolves these names
# LIVE against webtext's current state on every access instead, so `from pyutilz.text.strings import
# nlp` (or `pyutilz.text.strings.nlp`) always reflects whatever webtext.py has actually cached,
# preserving both the historic public surface AND live-state correctness.
# ----------------------------------------------------------------------------------------------------------------------------

_LAZY_WEBTEXT_GLOBALS = frozenset({"inflect_engine", "nlp", "ascii_emojies", "unicode_emojies"})

# The same set under the name the API-stability snapshot looks for (see webtext.py's declaration):
# this facade forwards these names live, so it inherits their order-dependent binding too.
__lazy_globals__ = _LAZY_WEBTEXT_GLOBALS

# Explicit public surface: makes this facade's re-export intent self-documenting (rather than
# relying on nothing but "this is __init__.py") and gives the self-scan code-audit
# possibly_dead_import check an authoritative usage signal for names only ever consumed by
# external repos (``from pyutilz.text.strings import X``), which its own-repo corpus scan can't
# see.
__all__ = [
    "logger",
    "ensure_installed",
    "Any", "Iterable", "List", "Optional", "Sequence", "Union",
    "is_float",
    "json_serial", "sub_elem", "jsonize_atrtributes", "remove_json_attributes",
    "leave_json_attributes", "extract_json_attribute", "remove_json_empty_attributes",
    "remove_json_defaults", "json_pg_dumps", "get_jsonlist_property", "get_jsonlist_properties",
    "read_config_file", "write_config_file",
    "find_between", "parse_tokens", "make_text_from_inner_html_elements", "underscorize_variable",
    "get_hash", "strip_characters", "strip_doubled_characters", "rpad", "shorten_path", "slugify",
    "camel_case_split",
    "punctuation", "eos", "spacy_sent_tokenize", "remove_videos", "fix_duplicate_tokens",
    "unescape_html", "fix_html", "parse_html", "fix_quotations", "fix_spaces",
    "fix_broken_sentences", "fix_missed_space_between_sentences", "merge_punctuation_signs",
    "ensure_space_after_comma", "clean_description", "get_ascii_emojies", "get_unicode_emojies",
    "sentencize_text", "suffixize",
    "tokenize_text", "tokenize_source", "tokenize_to_chars", "tokenize_to_words",
    "get_entropy_stats", "entropy", "entropy_rate", "compute_entropy_stats", "naive_entropy_rate",
    "stringify_dict",
    "inflect_engine", "nlp", "ascii_emojies", "unicode_emojies",  # kept in sync with _LAZY_WEBTEXT_GLOBALS above
]


def __getattr__(name):
    if name in _LAZY_WEBTEXT_GLOBALS:
        from . import webtext as _webtext

        return getattr(_webtext, name)
    if name in _LAZY_REEXPORTS:
        import importlib

        mod_name, attr = _LAZY_REEXPORTS[name]
        mod = importlib.import_module(mod_name)
        value = mod if attr is None else getattr(mod, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _LAZY_WEBTEXT_GLOBALS | set(_LAZY_REEXPORTS))

# ----------------------------------------------------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import doctest

    doctest.testmod()

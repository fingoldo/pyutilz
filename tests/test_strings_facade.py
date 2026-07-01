"""Sensor test for the strings subpackage split.

Asserts the `pyutilz.text.strings` facade re-exports a representative set of the historic public API and that
the re-exported objects are the *same* objects as those defined in the cohesive submodules (no accidental copies).
"""

import importlib

import pyutilz.text.strings as facade

# NB: import submodules by full path. The entropy submodule is deliberately named `textentropy` (not `entropy`)
# so it does not shadow the re-exported public function `entropy` on the package.
basics = importlib.import_module("pyutilz.text.strings.basics")
jsonutils = importlib.import_module("pyutilz.text.strings.jsonutils")
configfiles = importlib.import_module("pyutilz.text.strings.configfiles")
webtext = importlib.import_module("pyutilz.text.strings.webtext")
entropy = importlib.import_module("pyutilz.text.strings.textentropy")


def test_facade_reexports_representative_symbols():
    representative = [
        # jsonutils
        "json_serial", "extract_json_attribute", "json_pg_dumps", "get_jsonlist_property",
        # configfiles
        "read_config_file", "write_config_file",
        # basics
        "find_between", "slugify", "camel_case_split", "get_hash", "rpad", "shorten_path",
        # webtext
        "clean_description", "sentencize_text", "fix_html", "suffixize", "get_ascii_emojies", "eos", "punctuation",
        # entropy
        "tokenize_to_words", "compute_entropy_stats", "entropy", "stringify_dict",
    ]
    for name in representative:
        assert hasattr(facade, name), f"facade lost public symbol {name!r}"


def test_facade_symbols_are_same_objects_as_submodules():
    pairs = [
        (jsonutils, "json_serial"),
        (jsonutils, "json_pg_dumps"),
        (configfiles, "read_config_file"),
        (basics, "slugify"),
        (basics, "camel_case_split"),
        (webtext, "clean_description"),
        (webtext, "sentencize_text"),
        (webtext, "suffixize"),
        (entropy, "tokenize_to_words"),
        (entropy, "compute_entropy_stats"),
    ]
    for module, name in pairs:
        assert getattr(facade, name) is getattr(module, name), f"{name} on facade is not the submodule object"


def test_lazy_alias_still_resolves():
    import pyutilz.strings as alias

    assert alias.find_between("xAYb", "A", "b") == "Y"

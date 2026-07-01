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
# Normal Imports (kept for public-surface backward compatibility)
# ----------------------------------------------------------------------------------------------------------------------------

from typing import Any, Iterable, List, Optional, Sequence, Union

import pandas as pd, numpy as np
import string
import json

from collections import OrderedDict
import unicodedata
import re
import math
from collections import defaultdict, deque, Counter

from pyutilz.core.pythonlib import is_float

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
# These are stateful caches mutated (via `global`) inside webtext.py. The names below are re-exported here purely
# to preserve the historic public surface of the flat strings.py module (so `from pyutilz.text.strings import nlp`
# etc. keeps resolving). The authoritative, mutated copies live in webtext; read those for live state.
# ----------------------------------------------------------------------------------------------------------------------------

from .webtext import inflect_engine, nlp, ascii_emojies, unicode_emojies

# ----------------------------------------------------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import doctest

    doctest.testmod()

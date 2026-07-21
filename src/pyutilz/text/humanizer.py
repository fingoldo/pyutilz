"""De-LLM-ize text: strip AI-typical patterns and introduce realistic typos.

Usage::

    from pyutilz.text.humanizer import humanize

    text = humanize(ai_generated_text, typo_count=2)

The ``protected_spans`` parameter accepts ``(start, end)`` byte-offset pairs
that the typo engine must leave untouched (e.g. attention-check compliance
phrases in cover letters).
"""

from __future__ import annotations

import logging
import random
import re
from typing import Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# QWERTY adjacency maps
# ---------------------------------------------------------------------------

# Standard US QWERTY layout (lowercase).  Each key maps to its immediate
# physical neighbours (including diagonal).  Used by ``adjacent_key`` typos.
_QWERTY_ADJACENT: dict[str, str] = {
    "q": "wa", "w": "qeas", "e": "wrds", "r": "etdf", "t": "ryfg",
    "y": "tugh", "u": "yijh", "i": "uojk", "o": "iplk", "p": "ol",
    "a": "qwsz", "s": "wedxza", "d": "erfcxs", "f": "rtgvcd",
    "g": "tyhbvf", "h": "yujnbg", "j": "uikmnh", "k": "iolmj",
    "l": "opk",
    "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb",
    "b": "vghn", "n": "bhjm", "m": "njk",
}

# Russian ЙЦУКЕН layout adjacency.
_YCUKEN_ADJACENT: dict[str, str] = {
    "й": "цф", "ц": "йуфыв", "у": "цквыа", "к": "уеапр",
    "е": "кнпро", "н": "егрол", "г": "нщолд", "ш": "щзлдж",
    "щ": "гшздж", "з": "шщжхэ", "х": "зэж", "ъ": "хэ",
    "ф": "йцыя", "ы": "цуфвач", "в": "укыапс", "а": "кевпрм",
    "п": "еналри", "р": "нгпоит", "о": "гшрлдь", "л": "щгодж",
    "д": "шщлжэб", "ж": "зхдэю", "э": "хъжю",
    "я": "фыч", "ч": "ыясв", "с": "вчмап", "м": "аспи",
    "и": "прмт", "т": "ориьб", "ь": "олтбю", "б": "лдьюж",
    "ю": "джэб",
}


def _get_adjacent(char: str) -> str | None:
    """Return string of adjacent keys for *char*, or None if unsupported."""
    lower = char.lower()
    adj = _QWERTY_ADJACENT.get(lower) or _YCUKEN_ADJACENT.get(lower)
    if adj is None:
        return None
    # Preserve case of original character.
    if char.isupper():
        return adj.upper()
    return adj


# ---------------------------------------------------------------------------
# AI-pattern replacement tables
# ---------------------------------------------------------------------------

# Each entry is (compiled regex, replacement).
# Replacements are applied in order; use \b word boundaries to avoid
# false positives inside longer words.

_AI_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # --- Overused "filler" phrases (remove entirely) ---
    (re.compile(r"\bIt(?:'s| is) worth noting that\s*", re.I), ""),
    (re.compile(r"\bIt(?:'s| is) important to note that\s*", re.I), ""),
    (re.compile(r"\bIn conclusion,?\s*", re.I), ""),
    (re.compile(r"\bTo summarize,?\s*", re.I), ""),
    (re.compile(r"\bstands as a testament\b", re.I), "shows"),
    (re.compile(r"\bunderscore[sd]?\s+(?:its?|the)\s+importance\b", re.I), "matters"),
    (re.compile(r"\bAs an AI(?:\s+language model)?,?\s*", re.I), ""),
    # --- Vocabulary downgrades ---
    (re.compile(r"\bdelve(?:s|d)?\s+into\b", re.I), "look into"),
    (re.compile(r"\bdelve\b", re.I), "dig"),
    (re.compile(r"\bleverage(?:s|d)?\b", re.I), "use"),
    (re.compile(r"\butilize(?:s|d)?\b", re.I), "use"),
    (re.compile(r"\bfacilitate(?:s|d)?\b", re.I), "help with"),
    (re.compile(r"\bstreamline(?:s|d)?\b", re.I), "simplify"),
    (re.compile(r"\bcomprehensive\b", re.I), "thorough"),
    (re.compile(r"\bensure(?:s|d)?\s+that\b", re.I), "make sure"),
    (re.compile(r"\bensure\b", re.I), "make sure"),
    (re.compile(r"\bpivotal\b", re.I), "key"),
    (re.compile(r"\bseamless(?:ly)?\b", re.I), "smooth"),
    (re.compile(r"\brobust\b", re.I), "solid"),
    (re.compile(r"\bmeticulous(?:ly)?\b", re.I), "careful"),
    (re.compile(r"\binnovative\b", re.I), "new"),
    (re.compile(r"\bcutting[- ]edge\b", re.I), "modern"),
    # --- Transition word downgrades ---
    (re.compile(r"\bFurthermore,?\s*", re.I), "Also, "),
    (re.compile(r"\bMoreover,?\s*", re.I), "Also, "),
    (re.compile(r"\bAdditionally,?\s*", re.I), "Also, "),
    (re.compile(r"\bConsequently,?\s*", re.I), "So, "),
    (re.compile(r"\bNevertheless,?\s*", re.I), "Still, "),
    # --- Enthusiasm / self-promotion downgrades ---
    (re.compile(r"\bI(?:'m| am) (?:thrilled|excited) to\b", re.I), "I'm glad to"),
    (re.compile(r"\bI(?:'d| would) love to\b", re.I), "I'd like to"),
    (re.compile(r"\bI bring (\d+)\s+years? of\b", re.I), r"I have \1 years of"),
    (re.compile(r"\bplay(?:s|ed)?\s+a\s+(?:vital|crucial|pivotal|key)\s+role\b", re.I), "is important"),
    # --- Filler adverbs ---
    (re.compile(r"\bspecifically,?\s*", re.I), ""),
    (re.compile(r"\bessentially,?\s*", re.I), ""),
]


def strip_ai_patterns(text: str) -> str:
    """Replace common AI-generated phrases with more natural alternatives."""
    for pattern, replacement in _AI_PATTERNS:
        text = pattern.sub(replacement, text)
    # Clean up double spaces left by removals.
    text = re.sub(r"  +", " ", text)
    # Fix sentence starts that lost their capital after a removal.
    text = re.sub(r"(?<=\.\s)([a-zа-яё])", lambda m: m.group(1).upper(), text)
    return text.strip()


# ---------------------------------------------------------------------------
# Typography fixes
# ---------------------------------------------------------------------------

def fix_dashes(text: str) -> str:
    """Replace em/en-dashes with `` - `` and smart quotes with ASCII."""
    text = text.replace("\u2014", " - ")  # em-dash
    text = text.replace("\u2013", " - ")  # en-dash
    text = text.replace(" -- ", " - ")  # double-dash (AI marker)
    text = text.replace("\u201c", '"')  # left double smart quote
    text = text.replace("\u201d", '"')  # right double smart quote
    text = text.replace("\u2018", "'")  # left single smart quote
    text = text.replace("\u2019", "'")  # right single smart quote
    # Collapse runs of spaces around the replacement dashes.
    text = re.sub(r" {2,}", " ", text)
    return text


# ---------------------------------------------------------------------------
# Emoji removal
# ---------------------------------------------------------------------------

# Broad regex covering common emoji ranges + variation selectors + ZWJ.
_EMOJI_RE = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # misc symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map
    "\U0001f900-\U0001f9ff"  # supplemental symbols
    "\U0001fa00-\U0001fa6f"  # chess symbols
    "\U0001fa70-\U0001faff"  # symbols extended-A
    "\U00002600-\U000027bf"  # misc symbols + dingbats
    "\U0000fe00-\U0000fe0f"  # variation selectors
    "\U0000200d"  # ZWJ
    "\U00002702-\U000027b0"
    "\U0000231a-\U0000231b"
    "\U000023e9-\U000023f3"
    "\U000023f8-\U000023fa"
    "\U000025aa-\U000025ab"
    "\U000025b6"
    "\U000025c0"
    "\U000025fb-\U000025fe"
    "\U00002934-\U00002935"
    "\U00002b05-\U00002b07"
    "\U00002b1b-\U00002b1c"
    "\U00002b50"
    "\U00002b55"
    "\U00003030"
    "\U0000303d"
    "\U00003297"
    "\U00003299"
    "]+",
    flags=re.UNICODE,
)


def strip_emojis(text: str) -> str:
    """Remove emoji characters from *text*."""
    return _EMOJI_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Typo engine
# ---------------------------------------------------------------------------

# Typo types with weights (must sum to 1.0).
_TYPO_TYPES: list[tuple[str, float]] = [
    ("adjacent_key", 0.30),
    ("transpose", 0.20),
    ("double_letter", 0.15),
    ("skip_letter", 0.10),
    ("extra_space", 0.10),
    ("missing_space_comma", 0.10),
    ("missing_space_period", 0.05),
]

_TYPO_NAMES = [t[0] for t in _TYPO_TYPES]
_TYPO_WEIGHTS = [t[1] for t in _TYPO_TYPES]


def _is_protected(pos: int, length: int, protected: Sequence[tuple[int, int]]) -> bool:
    """Return True if the range [pos, pos+length) overlaps any protected span."""
    end = pos + length
    for ps, pe in protected:
        if pos < pe and end > ps:
            return True
    return False


def _pick_word_for_typo(
    words: list[tuple[int, str]],
    protected: Sequence[tuple[int, int]],
    rng: random.Random,
    min_len: int = 5,
) -> tuple[int, str] | None:
    """Pick a random word suitable for a character-level typo.

    Returns (start_offset, word) or None.
    Skips first/last words, short words, capitalised proper-noun-like words,
    and words overlapping protected spans.
    """
    candidates = []
    for i, (offset, word) in enumerate(words):
        if i == 0 or i == len(words) - 1:
            continue
        # Only alphabetic, long enough.
        if not word.isalpha() or len(word) < min_len:
            continue
        # Skip if it looks like a proper noun (capital + rest lower) in mid-sentence.
        if word[0].isupper() and word[1:].islower():
            continue
        if _is_protected(offset, len(word), protected):
            continue
        candidates.append((offset, word))
    if not candidates:
        return None
    return rng.choice(candidates)


def _apply_char_typo(
    text: str,
    typo_type: str,
    protected: Sequence[tuple[int, int]],
    rng: random.Random,
) -> str:
    """Apply a single character-level typo to *text*."""
    # Build word list with offsets.
    words: list[tuple[int, str]] = [(m.start(), m.group()) for m in re.finditer(r"\S+", text)]
    target = _pick_word_for_typo(words, protected, rng)
    if target is None:
        return text
    offset, word = target

    if typo_type == "adjacent_key":
        # Replace one character with an adjacent QWERTY/ЙЦУКЕН key.
        indices = [i for i in range(len(word)) if word[i].isalpha() and _get_adjacent(word[i])]
        if not indices:
            return text
        idx = rng.choice(indices)
        adj = _get_adjacent(word[idx])
        if adj is None:
            return text
        new_char = rng.choice(adj)
        new_word = word[:idx] + new_char + word[idx + 1 :]

    elif typo_type == "transpose":
        if len(word) < 3:
            return text
        idx = rng.randint(1, len(word) - 2)
        new_word = word[:idx] + word[idx + 1] + word[idx] + word[idx + 2 :]

    elif typo_type == "double_letter":
        idx = rng.randint(1, len(word) - 2)
        new_word = word[:idx] + word[idx] + word[idx:]

    elif typo_type == "skip_letter":
        idx = rng.randint(1, len(word) - 2)
        new_word = word[:idx] + word[idx + 1 :]

    else:
        return text

    return text[:offset] + new_word + text[offset + len(word) :]


def _apply_space_typo(
    text: str,
    typo_type: str,
    protected: Sequence[tuple[int, int]],
    rng: random.Random,
) -> str:
    """Apply a single spacing typo to *text*."""
    if typo_type == "extra_space":
        # Find multi-word gaps (single spaces between words).
        gaps = [m.start() for m in re.finditer(r"(?<=\S) (?=\S)", text)]
        gaps = [g for g in gaps if not _is_protected(g, 1, protected)]
        if not gaps:
            return text
        pos = rng.choice(gaps)
        return text[:pos] + "  " + text[pos + 1 :]

    elif typo_type == "missing_space_comma":
        # Find ", " patterns and remove the space.
        hits = [m.start() for m in re.finditer(r", ", text)]
        hits = [h for h in hits if not _is_protected(h, 2, protected)]
        if not hits:
            return text
        pos = rng.choice(hits)
        return text[: pos + 1] + text[pos + 2 :]  # remove space after comma

    elif typo_type == "missing_space_period":
        # Find ". X" patterns (sentence boundary) and remove the space.
        hits = [m.start() for m in re.finditer(r"\. [A-ZА-ЯЁ]", text)]
        hits = [h for h in hits if not _is_protected(h, 2, protected)]
        if not hits:
            return text
        pos = rng.choice(hits)
        return text[: pos + 1] + text[pos + 2 :]  # remove space after period

    return text


def introduce_typos(
    text: str,
    count: int = 1,
    rng: random.Random | None = None,
    protected_spans: Sequence[tuple[int, int]] = (),
) -> str:
    """Introduce *count* realistic typos into *text*.

    Parameters
    ----------
    text:
        Input text.
    count:
        Number of typos to introduce.
    rng:
        Optional seeded Random instance for reproducibility.
    protected_spans:
        Sequence of ``(start, end)`` character-offset pairs that must not be
        modified (e.g. attention-check compliance phrases).
    """
    if rng is None:
        rng = random.Random()  # nosec B311 - non-cryptographic text-variation sampling (typo injection), not security-sensitive
    protected = list(protected_spans)
    for _ in range(count):
        typo_type = rng.choices(_TYPO_NAMES, weights=_TYPO_WEIGHTS, k=1)[0]
        if typo_type in ("extra_space", "missing_space_comma", "missing_space_period"):
            text = _apply_space_typo(text, typo_type, protected, rng)
        else:
            text = _apply_char_typo(text, typo_type, protected, rng)
    return text


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def humanize(
    text: str,
    typo_count: int = 1,
    rng: random.Random | None = None,
    protected_spans: Sequence[tuple[int, int]] = (),
) -> str:
    """Full humanization pipeline.

    1. Strip AI-typical phrases.
    2. Fix em/en-dashes and smart quotes.
    3. Remove emoji.
    4. Introduce realistic typos (respecting *protected_spans*).

    Parameters
    ----------
    text:
        AI-generated text.
    typo_count:
        How many typos to inject (0 to skip).
    rng:
        Seeded Random for reproducibility.
    protected_spans:
        ``(start, end)`` pairs of character offsets computed against the ORIGINAL (pre-cleaning)
        ``text``. Each span's exact substring content is captured before the strip/fix stages
        run (which can insert/delete text and shift every downstream offset), then located by
        content in the post-cleaning text so the typo engine protects the same actual characters
        regardless of how much earlier stages shifted their position. If a protected substring's
        content itself gets altered by an earlier stage (rather than merely shifted), it can no
        longer be found and is logged (WARNING) and NOT protected -- silent partial protection
        would defeat the whole point of this parameter for a compliance-verification use case.
    """
    original_snippets = [text[s:e] for s, e in protected_spans]
    text = strip_ai_patterns(text)
    text = fix_dashes(text)
    text = strip_emojis(text)
    remapped_spans: list[tuple[int, int]] = []
    for snippet in original_snippets:
        if not snippet:
            continue
        idx = text.find(snippet)
        if idx == -1:
            logger.warning("humanize(): protected span content %r not found after cleaning -- not protected from typo injection.", snippet[:60])
            continue
        remapped_spans.append((idx, idx + len(snippet)))
    if typo_count > 0:
        text = introduce_typos(text, count=typo_count, rng=rng, protected_spans=remapped_spans)
    return text

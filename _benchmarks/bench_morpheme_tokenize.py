"""Bench for AdvancedTokenizer.tokenize()'s morpheme-length cap fix (2026-07-21 audit round 2, HIGH).

tokenize() used to enumerate EVERY substring of every word with no length cap: O(word_len**2)
substrings, each up to O(word_len) to slice/hash -> O(word_len**3) total per pathological word
(a long URL, a base64/JS blob that survives fix_html, any unbroken scraped-text run with no
whitespace). Bounding morpheme length to MAX_MORPHEME_LENGTH makes per-word cost linear in
word_len instead of cubic.

Measured (Python 3.14, best-of-3, this checkout -- isolated substring-enumeration loop only,
excludes the constant nltk sent/word-tokenize overhead neither variant touches):
    word_len=200:    old 0.0200s   new 0.0012s   ~16x faster
    word_len=400:    old 0.0798s   new 0.0026s   ~31x faster
    word_len=800:    old 0.5133s   new 0.0041s   ~124x faster
    word_len=1600:   old 3.5025s   new 0.0100s   ~351x faster
(speedup grows with word_len, consistent with cubic-vs-linear complexity)

Run: python _benchmarks/bench_morpheme_tokenize.py
"""

import time
from collections import defaultdict

from pyutilz.text.tokenizers import AdvancedTokenizer


def _old_word_pass(word: str) -> int:
    """Replicates the PRE-FIX unbounded (i, j) substring enumeration for one word (single word,
    single sentence: w=0, k=1) -- returns the number of morphemes processed."""
    word_len = len(word)
    num_occs: dict = defaultdict(int)
    count = 0
    for i in range(word_len):
        for j in range(1, word_len - i + 1):
            morpheme = word[i : i + j]
            base_morpheme = morpheme.lower()
            num_occs[base_morpheme] += 1
            count += 1
    return count


def _new_word_pass(word: str, max_morpheme_length: int = AdvancedTokenizer.MAX_MORPHEME_LENGTH) -> int:
    """Replicates the POST-FIX capped (i, j) substring enumeration for one word."""
    word_len = len(word)
    num_occs: dict = defaultdict(int)
    count = 0
    for i in range(word_len):
        max_j = min(word_len - i, max_morpheme_length)
        for j in range(1, max_j + 1):
            morpheme = word[i : i + j]
            base_morpheme = morpheme.lower()
            num_occs[base_morpheme] += 1
            count += 1
    return count


def _bench(word_len: int, calls: int, old: bool) -> float:
    word = "a" * word_len
    fn = _old_word_pass if old else _new_word_pass
    best = float("inf")
    for _ in range(calls):
        t = time.perf_counter()
        fn(word)
        best = min(best, time.perf_counter() - t)
    return best


if __name__ == "__main__":
    # Sanity check: for a SHORT word (<= MAX_MORPHEME_LENGTH), old and new produce the exact
    # same substring count -- the cap only changes behavior for words LONGER than the cap.
    short = "cat"
    assert _old_word_pass(short) == _new_word_pass(short)
    print("identity OK for short words (cap has no effect below MAX_MORPHEME_LENGTH)")

    print(f"{'word_len':>10} {'old (s)':>12} {'new (s)':>12} {'speedup':>10}")
    for n in (200, 400, 800, 1600):
        old_t = _bench(n, calls=3, old=True)
        new_t = _bench(n, calls=3, old=False)
        speedup = old_t / new_t if new_t > 0 else float("inf")
        print(f"{n:>10} {old_t:>12.4f} {new_t:>12.4f} {speedup:>9.0f}x")

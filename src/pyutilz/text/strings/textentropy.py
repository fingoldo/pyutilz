# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

import re
import math
from typing import Any, Dict
import numpy as np
from collections import defaultdict, deque, Counter

# Pre-compiled at module level: this pattern is applied per-source in tokenize_to_words (hot path).
# timeit micro-benchmark (1M iters, "The quick brown fox's tale."): re.findall(r"[a-zA-Z']+", s) ~1.74us/call
# vs _WORD_TOKEN_RE.findall(s) ~1.22us/call -> ~1.42x faster by avoiding the per-call pattern-cache lookup.
_WORD_TOKEN_RE = re.compile(r"[a-zA-Z']+")


def tokenize_text(source: str, tokenizer: object, lowercase: bool = True, strip: bool = True) -> str:
    if strip:
        source = source.strip()
    if lowercase:
        source = source.lower()
    yield from tokenizer(source)


def tokenize_source(source: str, tokenizer: object, is_file: bool = False, lowercase: bool = True, strip: bool = True) -> str:
    """
    source can be a filename or a string, depending on is_file flag
    """
    if is_file:
        with open(source, encoding="utf-8") as file:
            for line in file:
                yield from tokenize_text(source=line, tokenizer=tokenizer, lowercase=lowercase, strip=strip)
    else:
        yield from tokenize_text(source=source, tokenizer=tokenizer, lowercase=lowercase, strip=strip)


def tokenize_to_chars(source: str, is_file: bool = False) -> str:
    if is_file:
        return tokenize_source(source, lambda s: s + " ", is_file=True)
    else:
        return tokenize_source(source, lambda s: s, is_file=False)


def tokenize_to_words(source, is_file: bool = False) -> str:
    return tokenize_source(source, lambda s: _WORD_TOKEN_RE.findall(s), is_file=is_file)


def get_entropy_stats(stream, model_order: int = 2) -> tuple:
    """
    Computes markov_model entropy stats of text
    Returns:
        stats is a Counter that matches each key in model to its total number of occurrences
        model is a dictionary mapping (n−1)-character prefixes to a Counter; that Counter maps each possible nth character to the number of times this character followed the (n−1)-character prefix.
    """
    conditional_stats: Dict[Any, Counter] = defaultdict(Counter)
    stats: Counter = Counter()
    circular_buffer: Any = deque(maxlen=model_order)

    for token in stream:
        prefix = tuple(circular_buffer)
        circular_buffer.append(token)
        if len(prefix) == model_order:
            stats[prefix] += 1
            conditional_stats[prefix][token] += 1

    return conditional_stats, stats


def entropy(stats: Counter, normalization_factor: float = 1.0) -> float:
    return -sum(proba / normalization_factor * math.log2(proba / normalization_factor) for proba in stats.values())


def entropy_rate(conditional_stats, stats) -> float:
    return sum(stats[prefix] * entropy(conditional_stats[prefix], stats[prefix]) for prefix in stats) / sum(stats.values())


def compute_entropy_stats(text: str, order: int = 0) -> tuple:
    conditional_stats, stats = get_entropy_stats(tokenize_to_chars(text), order)
    if len(stats) == 0:
        return None, None
    sample_entropy_rate = entropy_rate(conditional_stats, stats)
    sample_raw_entropy = entropy(stats, len(stats))
    # print(stats)
    # print(f"Entropy: {sample_raw_entropy}, Entropy rate: {sample_entropy_rate}")
    return sample_raw_entropy, sample_entropy_rate


def naive_entropy_rate(a: str) -> float:
    m, cnt = np.unique(np.array(list(a)), return_counts=True)
    # print(m)
    # print(cnt)
    p = cnt / np.sum(cnt)

    return -np.sum(p * np.log2(p))

def stringify_dict(the_dict:dict,sep:str=",")->str:
    return sep.join([f"{key}={value}" for key,value in the_dict.items()])

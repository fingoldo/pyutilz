"""(internal) part of pyutilz.dev.code_audit; see package __init__ for docs."""
from __future__ import annotations

import ast
import difflib
import re
from collections import defaultdict
from pathlib import Path

from ._base import Finding, _DEFAULT_EXCLUDE_DIRS, _iter_py_files, _safe_parse, _line_text
from .duplicate_function_body import _stripped_body, _node_count, _normalized_body

# Fuzzy matching a trivial body (a short property getter, a one-line delegate) produces spurious
# high-ratio hits between structurally-unrelated functions purely because there isn't enough text
# for the ratio to discriminate on. Set well above scan_duplicate_function_body's exact-match floor
# of 8: exactness has no such noise floor, similarity does.
_MIN_BODY_NODES = 20

_DEFAULT_MIN_SIMILARITY = 0.99  # overall-similarity threshold: comparable length AND content
_DEFAULT_MIN_CONTAINMENT = 0.98  # subset threshold: the SHORTER body appears ~wholly inside the longer one

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[^\sA-Za-z0-9_]")

# Candidate-generation shingle size, in tokens of the structural AST dump (not source tokens).
# 8 AST-dump tokens is specific enough (node type + field names + a couple of operators/values)
# that two functions sharing one is already a strong signal, not noise.
_SHINGLE_SIZE = 8

# A function pair must share at least this FRACTION of the SHORTER function's own shingle
# count before paying for the exact SequenceMatcher pass. Using min(len_i, len_j) as the
# denominator (not the union, as a Jaccard-style estimate would) matters: it is what makes
# this threshold work for BOTH detection modes below -- for two comparable-length functions it
# approximates their overall similarity, and for a small function embedded in a much larger
# one it approximates CONTAINMENT (the union-based Jaccard alternative was tried and rejected:
# it dilutes a small function's shingles into irrelevance against a much bigger superset's
# total, systematically missing the containment case this scanner exists to catch). A genuine
# >=99%-similar or >=98%-contained pair clears this easily; it exists purely to reject the long
# tail of pairs sharing a couple of shingles by chance (confirmed in the wild: a flat absolute
# count threshold of 2 on a ~2000-function real corpus produced 116830 "candidate" pairs,
# almost all noise, each still paying for a real SequenceMatcher call). This is a
# candidate-generation floor, never a correctness filter -- every pair that passes it still
# goes through the real ratio/containment check.
_MIN_SHARED_SHINGLE_FRACTION = 0.5

# A shingle shared by more functions than this is generic boilerplate (a common control-flow
# shape reproduced independently across many unrelated functions), dropped from the index
# purely to bound candidate-generation work. Keep this MODERATE, not aggressive: too low is a
# correctness bug (confirmed in the wild -- a cap of 25 silently dropped true-positive pairs in
# unrelated production files depending on whether tests/ was scanned alongside them, since
# common test-fixture shingles could push a shared production shingle's count over a too-low
# cap), while too high reintroduces the O(bucket_size^2) blowup the cap exists to prevent (also
# confirmed in the wild at a cap of 250). This is a candidate-generation backstop, not the
# primary noise filter -- `_MIN_SHARED_SHINGLE_FRACTION` above does that job; a shingle common
# enough to hit this cap is common enough that dropping it costs negligible recall on a
# genuine near-duplicate/subset pair, which still shares plenty of other, more specific shingles.
_MAX_POSTING_LIST = 80

# difflib.SequenceMatcher over TOKEN sequences (not raw source) is adversarial for very large
# functions: a structural AST dump is dominated by a small vocabulary of repeated tokens
# ("Name", "Load", "Store", punctuation), so `find_longest_match`'s internal b2j index entries
# grow into long lists precisely for the elements it has to scan most -- confirmed in the wild,
# a handful of candidate pairs among functions of a few thousand tokens each took tens of
# seconds apiece with `autojunk=False` (deliberately off so a real match isn't miscounted as
# junk). A function large enough to hit this cap already has bigger problems a size/complexity
# check should flag directly; treating it as out of scope here trades a small amount of recall
# on enormous functions for a hard bound on worst-case per-pair cost.
_MAX_TOKENS_FOR_EXACT_CHECK = 3000


def _tokenize(normalized: str) -> tuple[str, ...]:
    """Split a structural AST dump into identifier/punctuation tokens.

    Comparing token LISTS instead of raw characters cuts the sequence length
    difflib.SequenceMatcher has to work over by roughly the average token
    width (~5-8x for typical AST-dump text), which is what makes running it
    over thousands of function bodies tractable at all.
    """
    return tuple(_TOKEN_RE.findall(normalized))


def _is_nested(outer: "ast.FunctionDef | ast.AsyncFunctionDef", inner: "ast.FunctionDef | ast.AsyncFunctionDef") -> bool:
    """True if ``inner`` is lexically defined somewhere inside ``outer``'s body (a closure,
    a decorator factory's builder function, ...). Only meaningful for two nodes from the SAME
    file/tree; the caller is responsible for that check."""
    if outer is inner:
        return False
    return any(child is inner for child in ast.walk(outer))


def _shingles(tokens: tuple[str, ...], k: int = _SHINGLE_SIZE) -> set[int]:
    """The set of hashed overlapping k-token windows in ``tokens`` (or a single hash of the
    whole tuple if it's shorter than ``k``, or empty for an empty tuple)."""
    if len(tokens) < k:
        return {hash(tokens)} if tokens else set()
    return {hash(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def scan_near_duplicate_function_body(
    root: Path,
    exclude_dirs: frozenset[str] = _DEFAULT_EXCLUDE_DIRS,
    min_nodes: int = _MIN_BODY_NODES,
    min_similarity: float = _DEFAULT_MIN_SIMILARITY,
    min_containment: float = _DEFAULT_MIN_CONTAINMENT,
) -> list[Finding]:
    """Find functions/methods whose bodies are NEARLY (but not exactly) identical, in
    either of two shapes:

    1. **Near-duplicate** (``Finding.check == "near_duplicate_function_body"``): two bodies
       of comparable length whose overall ``difflib.SequenceMatcher`` ratio over the
       structural AST-dump tokens (the same normalization ``scan_duplicate_function_body``
       uses for its exact-match check) is ``>= min_similarity``. Catches a copy-paste that
       then drifted by one renamed variable, one changed constant, or one extra guard clause.

    2. **Subset** (``Finding.check == "duplicate_function_body_subset"``): the SHORTER of two
       bodies appears ~wholly (``>= min_containment`` of its tokens, by matched-block length)
       inside the LONGER one, regardless of how much extra code the longer one has around it.
       A plain overall-similarity ratio would miss this: ratio normalizes by the COMBINED
       length of both sequences, so a 20-node helper fully duplicated inside a 200-node
       function scores a low overall ratio despite being a complete, verbatim copy. Catches
       "someone inlined a whole existing helper's logic instead of calling it."

    Both exclude EXACT matches (``scan_duplicate_function_body``'s job) and dunder methods
    (legitimate protocol-shape convergence, not drift). Candidate pairs are found via a
    token-shingle inverted index rather than an all-pairs sweep -- comparing every function to
    every other function directly is O(n^2) exact-match calls over already-large token
    sequences, intractable past a few hundred functions; see the module-level comments on
    ``_MIN_SHARED_SHINGLE_FRACTION``, ``_MAX_POSTING_LIST``, and ``_MAX_TOKENS_FOR_EXACT_CHECK``
    for how candidate generation stays both fast and correct across different ``exclude_dirs``.

    ``min_nodes`` guards against fuzzy-matching trivial bodies: a reasonable minimum length
    is what makes a high similarity/containment score a meaningful signal instead of two
    unrelated one-liners (or one tiny helper embedded in countless unrelated functions)
    coincidentally looking alike.

    Severity: Low for both shapes -- unlike an exact duplicate, either might be two
    independently evolved implementations that happen to still look alike, or a deliberate
    inlining; a human should judge whether to unify before acting on this.

    Known limitation: candidate generation is a bounded heuristic (see ``_MAX_POSTING_LIST``),
    so which pairs get reported can shift slightly depending on what ELSE is in the scanned
    tree -- confirmed in the wild, scanning a large test suite alongside production code can
    push a shared-but-informative shingle's global count over the cap and drop a production
    pair that a narrower scan would have found. This project's own convention (see
    ``test_code_audit_baseline.py``) is to exclude ``tests/`` from this class of scanner for
    exactly this reason; a fully cap-free, always-consistent scan would need true O(n^2)
    exact comparison, which is intractable at real-codebase scale (see the same constant's
    comment for the concrete blowups measured without it).
    """
    entries: list[tuple[Path, "ast.FunctionDef | ast.AsyncFunctionDef", list[str], tuple[str, ...]]] = []
    for py in _iter_py_files(root, exclude_dirs):
        tree = _safe_parse(py)
        if tree is None:
            continue
        src_lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("__") and node.name.endswith("__"):
                continue
            body = _stripped_body(node)
            if _node_count(body) < min_nodes:
                continue
            normalized = _normalized_body(body)
            if not normalized:
                continue
            tokens = _tokenize(normalized)
            if not tokens:
                continue
            entries.append((py, node, src_lines, tokens))

    shingle_index: dict[int, list[int]] = defaultdict(list)
    entry_shingles: list[set[int]] = []
    for idx, (_py, _node, _src, tokens) in enumerate(entries):
        sh = _shingles(tokens)
        entry_shingles.append(sh)
        for h in sh:
            shingle_index[h].append(idx)

    generic_shingles = {h for h, postings in shingle_index.items() if len(postings) > _MAX_POSTING_LIST}
    for h in generic_shingles:
        del shingle_index[h]

    candidate_pairs: set[tuple[int, int]] = set()
    for idx, sh in enumerate(entry_shingles):
        if not sh:
            continue
        shared_counts: dict[int, int] = defaultdict(int)
        for h in sh:
            for other in shingle_index.get(h, ()):
                if other != idx:
                    shared_counts[other] += 1
        for other, count in shared_counts.items():
            threshold = _MIN_SHARED_SHINGLE_FRACTION * min(len(sh), len(entry_shingles[other]))
            if count >= threshold:
                candidate_pairs.add((idx, other) if idx < other else (other, idx))

    findings: list[Finding] = []
    for i, j in sorted(candidate_pairs):
        py_i, node_i, src_i, tokens_i = entries[i]
        py_j, node_j, src_j, tokens_j = entries[j]
        if tokens_i == tokens_j:
            continue  # exact duplicate: scan_duplicate_function_body's job, not ours
        if max(len(tokens_i), len(tokens_j)) > _MAX_TOKENS_FOR_EXACT_CHECK:
            continue
        if py_i == py_j and (_is_nested(node_i, node_j) or _is_nested(node_j, node_i)):
            # A nested def is trivially "contained" in its enclosing function's AST -- that's
            # not copy-paste, it's just lexical scope. Confirmed in the wild: this was the
            # dominant false-positive source before the check existed (e.g. a closure factory
            # whose inner builder function got reported as "inlining" the outer function).
            continue

        matcher = difflib.SequenceMatcher(None, tokens_i, tokens_j, autojunk=False)
        ratio = matcher.ratio()
        matched = sum(block.size for block in matcher.get_matching_blocks())
        len_i, len_j = len(tokens_i), len(tokens_j)
        shorter_len = min(len_i, len_j)
        containment = (matched / shorter_len) if shorter_len else 0.0

        rel_i = py_i.relative_to(root).as_posix()
        rel_j = py_j.relative_to(root).as_posix()

        if ratio >= min_similarity:
            findings.append(
                Finding(
                    check="near_duplicate_function_body",
                    severity="Low",
                    file=rel_j,
                    line=node_j.lineno,
                    snippet=_line_text(src_j, node_j.lineno),
                    detail=(
                        f"def {node_j.name}(...) is {ratio:.0%} structurally similar to "
                        f"{rel_i}:{node_i.lineno} (def {node_i.name}) -- likely copy-paste that "
                        "then drifted; consider unifying into one shared function, or confirm "
                        "these are genuinely independent implementations that happen to look alike."
                    ),
                )
            )
        elif containment >= min_containment:
            # Whichever body is SHORTER is the one (near-)wholly embedded in the longer one.
            if len_i <= len_j:
                sub_node, sub_rel = node_i, rel_i
                sup_node, sup_src, sup_rel = node_j, src_j, rel_j
            else:
                sub_node, sub_rel = node_j, rel_j
                sup_node, sup_src, sup_rel = node_i, src_i, rel_i
            findings.append(
                Finding(
                    check="duplicate_function_body_subset",
                    severity="Low",
                    file=sup_rel,
                    line=sup_node.lineno,
                    snippet=_line_text(sup_src, sup_node.lineno),
                    detail=(
                        f"def {sup_node.name}(...) contains {containment:.0%} of "
                        f"{sub_rel}:{sub_node.lineno} (def {sub_node.name})'s body verbatim -- "
                        "looks like that helper's whole logic was inlined here instead of called; "
                        "consider extracting the shared part into one function both use."
                    ),
                )
            )
    return findings

"""Meta-test proposed after the 2026-07-21 audit: pandaslib and polarslib implement conceptually
equivalent operations (e.g. "drop constant columns", "downcast float64 to float32") under DIFFERENT
names and, critically, DIFFERENT mutation contracts -- pandaslib idiomatically mutates the frame
in place and returns None (or the list of dropped/kept column names), polarslib's DataFrame is
functional/immutable so its sibling MUST return the transformed frame. A caller porting pipeline
code from the pandas backend to the polars backend (or vice versa) who keeps the pandas
"call it, ignore the return value" calling convention against the polars sibling silently drops
the entire transformation -- the exact class of bug this test exists to catch BEFORE it reaches a
real pipeline, by pinning each known pair's contract and failing loudly if either side drifts.

Deliberately a manually-curated pair list, not name-similarity auto-detection: an automated
"fuzzy-match function names across the two modules" pass was assessed as producing too many false
positives (many pandaslib/polarslib functions are similarly named but NOT semantically paired,
e.g. general dtype helpers). Add a new entry here whenever a new intentionally-paired
pandaslib/polarslib operation is introduced.
"""
from __future__ import annotations

import inspect

import pyutilz.data.pandaslib as pandaslib
import pyutilz.data.polarslib as polarslib

# Each entry: (pandaslib_name, polarslib_name, pandaslib_mutates_in_place, polarslib_mutates_in_place)
# "mutates_in_place" here means: the function's PRIMARY contract is "call it for the side effect,
# the return value is secondary/None" (pandas idiom) vs "the return value IS the result, the input
# frame is never modified" (polars idiom, since polars.DataFrame columns can't be dropped in
# place the way pandas' can).
_SIBLING_PAIRS = [
    ("remove_constant_columns", "drop_constant_columns", True, False),
    ("convert_float64_to_float32", "cast_f64_to_f32", False, False),
]


def test_sibling_pairs_still_exist():
    """Both sides of every declared sibling pair must still be importable -- a rename on either
    side without updating this pin (or the other side) is exactly the drift this test guards."""
    missing = []
    for pd_name, pl_name, _, _ in _SIBLING_PAIRS:
        if not hasattr(pandaslib, pd_name):
            missing.append(f"pandaslib.{pd_name}")
        if not hasattr(polarslib, pl_name):
            missing.append(f"polarslib.{pl_name}")
    assert not missing, f"sibling pair member(s) no longer exist -- update _SIBLING_PAIRS or restore them: {missing}"


def test_sibling_pairs_mutation_contract_matches_pin():
    """Pin each pair's pandas-idiom-vs-functional mutation contract via its return annotation
    (``-> None`` == in-place idiom; anything else == functional/returns-the-result). A change to
    either side's return annotation without a corresponding, deliberate update here is exactly the
    "caller silently drops the polars transformation" bug class this test exists to catch."""
    mismatches = []
    for pd_name, pl_name, pd_expected_inplace, pl_expected_inplace in _SIBLING_PAIRS:
        pd_fn = getattr(pandaslib, pd_name)
        pl_fn = getattr(polarslib, pl_name)
        pd_sig = inspect.signature(pd_fn)
        pl_sig = inspect.signature(pl_fn)
        pd_actual_inplace = pd_sig.return_annotation is None
        pl_actual_inplace = pl_sig.return_annotation is None
        if pd_actual_inplace != pd_expected_inplace:
            mismatches.append(f"pandaslib.{pd_name}: pinned mutates_in_place={pd_expected_inplace}, " f"actual return annotation={pd_sig.return_annotation!r}")
        if pl_actual_inplace != pl_expected_inplace:
            mismatches.append(f"polarslib.{pl_name}: pinned mutates_in_place={pl_expected_inplace}, " f"actual return annotation={pl_sig.return_annotation!r}")
    assert not mismatches, (
        "sibling pair mutation contract drifted from the pin in _SIBLING_PAIRS -- if this is an "
        "intentional, reviewed change, update the pin; if not, a caller porting pipeline code "
        "between the pandas and polars backend can now silently drop a transformation:\n  " + "\n  ".join(mismatches)
    )


def test_sibling_pairs_functional_side_actually_returns_a_frame():
    """For any pair member pinned as NOT in-place (functional), its return annotation must name an
    actual frame-like type (not bare ``None``, not unannotated) -- confirms the "call it and use
    the return value" contract is real, not just "the annotation happens not to be None"."""
    unclear = []
    for pd_name, pl_name, pd_expected_inplace, pl_expected_inplace in _SIBLING_PAIRS:
        if not pd_expected_inplace:
            sig = inspect.signature(getattr(pandaslib, pd_name))
            if sig.return_annotation is inspect.Signature.empty:
                unclear.append(f"pandaslib.{pd_name} has no return annotation at all")
        if not pl_expected_inplace:
            sig = inspect.signature(getattr(polarslib, pl_name))
            if sig.return_annotation is inspect.Signature.empty:
                unclear.append(f"polarslib.{pl_name} has no return annotation at all")
    assert not unclear, f"functional sibling(s) with no return annotation to verify the contract against: {unclear}"

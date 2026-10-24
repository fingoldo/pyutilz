import typing
import pytest


def test_f05_add_weighted_aggregates_declares_a_selector():
    """`object` supports no `-`; the signature type-checked only because the other operand was Any."""
    cs = pytest.importorskip("polars.selectors")
    if not hasattr(cs, "Selector"):
        pytest.skip("this polars has no cs.Selector; the annotation is deliberately a string for that reason")
    from pyutilz.data.polarslib.aggregations import add_weighted_aggregates

    # Resolved, not compared as source text: the annotation is quoted so importing this module
    # does not require a polars new enough to define cs.Selector, but it must still resolve to it.
    hints = typing.get_type_hints(add_weighted_aggregates)
    assert hints["columns_selector"] is cs.Selector

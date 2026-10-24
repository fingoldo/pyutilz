def test_f12_get_attr_accepts_none_and_returns_the_default():
    """The `unwanted_value` guard exists to catch exactly this; `obj: dict` said it could not fire."""
    from pyutilz.core.pythonlib.objects import get_attr

    assert get_attr(None, "hosts") == []
    assert get_attr(None, "hosts", default_value="x") == "x"
    # A non-None unwanted_value used to walk straight into None.get(...).
    assert get_attr(None, "hosts", unwanted_value="sentinel") == []
    assert get_attr({"hosts": [1, 2]}, "hosts") == [1, 2]


def test_f12_get_attr_return_is_not_object():
    from pyutilz.core.pythonlib.objects import get_attr

    assert get_attr.__annotations__["return"] is not object

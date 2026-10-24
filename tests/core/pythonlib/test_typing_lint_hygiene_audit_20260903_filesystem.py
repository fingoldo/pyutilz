def test_f08_objects_dumper_process_object_returns_false_not_none(tmp_path):
    """`process_objects` counts these results; None worked only by being falsy."""
    from pyutilz.core.pythonlib.filesystem import ObjectsDumper, ObjectsLoader

    dumper = ObjectsDumper()
    assert dumper._process_object({"a": None}, "a", str(tmp_path / "a.dump")) is False
    loader = ObjectsLoader()
    assert loader._process_object({}, "a", str(tmp_path / "missing.dump")) is False

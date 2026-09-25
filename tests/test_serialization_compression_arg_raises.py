"""serialize()/unserialize() reject a bad compression level with an exception that survives ``python -O``."""

import subprocess
import sys

import pytest

from pyutilz.core.serialization import serialize, unserialize


@pytest.mark.parametrize("fn", [lambda c: serialize({"a": 1}, compression=c), lambda c: unserialize(b"x", compression=c)])
def test_bad_compression_raises(fn):
    with pytest.raises(TypeError):
        fn("9")
    with pytest.raises(ValueError):
        fn(10)


def test_the_check_survives_optimised_mode():
    code = "from pyutilz.core.serialization import serialize\ntry:\n    serialize({'a': 1}, compression=42)\nexcept ValueError:\n    print('raised')\n"
    out = subprocess.run([sys.executable, "-O", "-c", code], capture_output=True, text=True, timeout=300)
    assert out.stdout.strip() == "raised", out.stderr[-1000:]

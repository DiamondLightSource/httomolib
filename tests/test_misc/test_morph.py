import numpy as np
import pytest

from httomolib.misc.morph import data_reducer

cupy_enabled = True
try:
    import cupy as xp

    try:
        xp.cuda.Device(0).compute_capability

    except xp.cuda.runtime.CUDARuntimeError:
        import numpy as xp

        cupy_enabled = False

except ImportError:

    import numpy as xp

    cupy_enabled = False


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("method", ["mean", "median"])
def test_data_reducer(host_data, method, axis):
    if cupy_enabled:
        reduced_data = data_reducer(xp.asarray(host_data), axis=axis, method=method)
        reduced_data = reduced_data.get()
    else:
        reduced_data = data_reducer(host_data, axis=axis, method=method)

    assert reduced_data.flags.c_contiguous
    assert reduced_data.dtype == np.float32
    if axis == 0:
        assert reduced_data.shape == (1, 128, 160)
    if axis == 1:
        assert reduced_data.shape == (180, 1, 160)
    if axis == 2:
        assert reduced_data.shape == (180, 128, 1)

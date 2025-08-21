from typing import Literal
import numpy as np
import pytest

from httomolib.misc.rescale import rescale_to_int


def test_rescale_no_change():
    data = np.random.randint(0, 255, size=(50, 10, 50), dtype=np.uint8).astype(
        np.float32
    )

    res_cpu = rescale_to_int(data, bits=8, glob_stats=(0.0, 255.0, 100.0, data.size))

    assert res_cpu.dtype == np.uint8
    np.testing.assert_array_equal(res_cpu, data)


@pytest.mark.parametrize("bits", [8, 16, 32])
def test_rescale_no_change_no_stats(bits: Literal[8, 16, 32]):
    data = np.ones((30, 10, 50), dtype=np.float32)
    data[0, 0] = 0.0
    data[13, 1] = (2**bits) - 1

    res_cpu = rescale_to_int(data, bits=bits)

    assert res_cpu.dtype.itemsize == bits // 8


def test_rescale_double():
    data = np.ones((30, 5, 50), dtype=np.float32)

    res_cpu = rescale_to_int(data, bits=8, glob_stats=(0, 2, 100, data.size))

    np.testing.assert_array_almost_equal(res_cpu, 127.0)


def test_rescale_double_offset():
    data = np.ones((30, 10, 50), dtype=np.float32) + 10

    res_cpu = rescale_to_int(data, bits=8, glob_stats=(10, 12, 100, data.size))

    np.testing.assert_array_almost_equal(res_cpu, 127.0)


@pytest.mark.parametrize("bits", [8, 16])
def test_rescale_double_offset_min_percentage(bits: Literal[8, 16, 32]):
    data = np.ones((30, 10, 50), dtype=np.float32) + 15
    data[0, 0, 0] = 10
    data[0, 0, 1] = 20

    res_cpu = rescale_to_int(
        data,
        bits=bits,
        glob_stats=(10, 20, 100, data.size),
        perc_range_min=10.0,
        perc_range_max=90.0,
    )

    max = (2**bits) - 1
    num = int(5 / 8 * max)
    # note: with 32bit, the float type actually overflows and the result is not full precision
    res_cpu = res_cpu.astype(np.float32)
    np.testing.assert_array_almost_equal(res_cpu[1:, :, :], num)
    assert res_cpu[0, 0, 0] == 0.0
    assert res_cpu[0, 0, 1] == max


def test_tomo_data_scale(host_data):

    res_cpu = rescale_to_int(
        host_data.astype(np.float32), perc_range_min=10, perc_range_max=90, bits=8
    )

    assert res_cpu.dtype == np.uint8

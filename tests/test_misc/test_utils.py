import numpy as np
import pytest

from numpy.testing import assert_equal

from httomolib.misc.utils import data_checker, __zeros_check


def test_data_checker_naninfs_uint16_1():
    data_input = np.ones(shape=(10, 10, 10)).astype(np.uint16) * 100
    data_output = data_checker(data_input.copy(), infsnans_correct=True)

    assert data_output.dtype == data_input.dtype
    assert data_output.shape == (10, 10, 10)


def test_data_checker_naninfs_float32_1():
    data_input = np.random.randint(0, 255, size=(10, 10, 10), dtype=np.uint8).astype(
        np.float32
    )
    data_input[1, 1, 1] = -np.inf
    data_input[1, 1, 2] = np.inf
    data_input[1, 1, 3] = np.nan

    data_output = data_checker(data_input.copy(), infsnans_correct=True)

    assert_equal(
        data_output[1, 1, 1],
        0,
    )
    assert_equal(
        data_output[1, 1, 2],
        0,
    )
    assert_equal(
        data_output[1, 1, 3],
        0,
    )
    assert data_output.dtype == data_output.dtype
    assert data_output.shape == (10, 10, 10)


@pytest.mark.parametrize("datatype", [np.uint16, np.float32])
def test_zeros_1(datatype):
    data_input = np.ones(shape=(10, 10, 10), dtype=datatype) * 100
    zeros_count = __zeros_check(
        data_input.copy(), verbosity=True, percentage_threshold=50, method_name=None
    )

    assert zeros_count == 0


@pytest.mark.parametrize("datatype", [np.uint16, np.float32])
def test_zeros_2(datatype):
    data_input = np.ones(shape=(10, 10, 10), dtype=datatype) * 100
    data_input[2:7, 1, 1] = 0
    zeros_count = __zeros_check(
        data_input.copy(), verbosity=True, percentage_threshold=50, method_name=None
    )

    assert zeros_count == 5


@pytest.mark.parametrize("datatype", [np.uint16, np.float32])
def test_zeros_3(datatype):
    data_input = np.ones(shape=(10, 10, 10), dtype=datatype) * 100
    data_input[2:7, :, :] = 0
    zeros_count = __zeros_check(
        data_input.copy(), verbosity=True, percentage_threshold=50, method_name=None
    )

    assert zeros_count == 500
